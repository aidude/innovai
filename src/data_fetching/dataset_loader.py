"""
Creativity Dataset Loader
Loads and processes the adversarial creativity training dataset
"""

import json
import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from pathlib import Path
from sentence_transformers import SentenceTransformer
from logging_config import setup_logging
logger = setup_logging(module_name=__name__)



@dataclass
class CreativityExample:
    """Single creativity training example"""
    id: str
    context: str
    predictable: str
    creative: str
    meta_creative: str
    surprise_source: str
    domain: str
    category: str  # 'historical_events' or 'visual_descriptions'

class CreativityDatasetLoader:
    """Loads and processes the creativity training dataset"""
    
    def __init__(self, dataset_path: str = "../../prompts/creativity_training_dataset.json"):
        # Convert relative path to absolute path from the current file's location
        current_dir = Path(__file__).parent
        self.dataset_path = (current_dir / dataset_path).resolve()
        self.dataset = None
        self.examples = []
        self.sentence_transformer = None
        logger.info(f"Initializing dataset loader with path: {self.dataset_path}")
        
    def load_dataset(self) -> Dict:
        """Load the JSON dataset from file"""
        try:
            with open(self.dataset_path, 'r', encoding='utf-8') as f:
                self.dataset = json.load(f)
            
            logger.info(f"Loaded dataset: {self.dataset['dataset_info']['name']}")
            logger.info(f"Version: {self.dataset['dataset_info']['version']}")
            logger.info(f"Total examples: {self.dataset['dataset_info']['total_examples']}")
            
            return self.dataset
            
        except FileNotFoundError:
            logger.error(f"Dataset file not found: {self.dataset_path}")
            raise
        except json.JSONDecodeError as e:
            logger.error(f"Invalid JSON in dataset file: {e}")
            raise
    
    def parse_examples(self) -> List[CreativityExample]:
        """Parse dataset into CreativityExample objects"""
        if not self.dataset:
            self.load_dataset()
        
        self.examples = []
        
        # Parse historical events
        for item in self.dataset['historical_events']:
            example = CreativityExample(
                id=item['id'],
                context=item['context'],
                predictable=item['predictable'],
                creative=item['creative'],
                meta_creative=item['meta_creative'],
                surprise_source=item['surprise_source'],
                domain=item['domain'],
                category='historical_events'
            )
            self.examples.append(example)
        
        # Parse visual descriptions
        for item in self.dataset['visual_descriptions']:
            example = CreativityExample(
                id=item['id'],
                context=item['context'],
                predictable=item['predictable'],
                creative=item['creative'],
                meta_creative=item['meta_creative'],
                surprise_source=item['surprise_source'],
                domain=item.get('visual_elements', ['visual'])[0],  # Use first visual element as domain
                category='visual_descriptions'
            )
            self.examples.append(example)
        
        logger.info(f"Parsed {len(self.examples)} examples")
        return self.examples
    
    def get_training_splits(self, train_ratio: float = 0.8) -> Tuple[List[CreativityExample], List[CreativityExample]]:
        """Split examples into training and validation sets"""
        if not self.examples:
            self.parse_examples()
        
        # Shuffle examples
        np.random.seed(42)  # For reproducibility
        shuffled_examples = self.examples.copy()
        np.random.shuffle(shuffled_examples)
        
        # Split
        split_idx = int(len(shuffled_examples) * train_ratio)
        train_examples = shuffled_examples[:split_idx]
        val_examples = shuffled_examples[split_idx:]
        
        logger.info(f"Training examples: {len(train_examples)}")
        logger.info(f"Validation examples: {len(val_examples)}")
        
        return train_examples, val_examples
    
    def get_predictor_training_data(self, examples: Optional[List[CreativityExample]] = None) -> List[Tuple[str, str]]:
        """Get context-predictable pairs for training predictors"""
        if examples is None:
            examples = self.examples if self.examples else self.parse_examples()
        
        predictor_data = []
        for example in examples:
            predictor_data.append((example.context, example.predictable))
        
        return predictor_data
    
    def get_generator_training_data(self, examples: Optional[List[CreativityExample]] = None, 
                                  creativity_level: str = 'creative') -> List[Tuple[str, str]]:
        """Get context-creative pairs for training generator"""
        if examples is None:
            examples = self.examples if self.examples else self.parse_examples()
        
        generator_data = []
        for example in examples:
            if creativity_level == 'creative':
                target = example.creative
            elif creativity_level == 'meta_creative':
                target = example.meta_creative
            else:
                target = example.predictable
            
            generator_data.append((example.context, target))
        
        return generator_data
    
    def get_surprise_analysis_data(self, examples: Optional[List[CreativityExample]] = None) -> pd.DataFrame:
        """Get data for analyzing surprise sources and patterns"""
        if examples is None:
            examples = self.examples if self.examples else self.parse_examples()
        
        data = []
        for example in examples:
            data.append({
                'id': example.id,
                'category': example.category,
                'domain': example.domain,
                'surprise_source': example.surprise_source,
                'context': example.context,
                'predictable': example.predictable,
                'creative': example.creative,
                'meta_creative': example.meta_creative
            })
        
        return pd.DataFrame(data)
    
    def compute_embeddings(self, model_name: str = 'all-MiniLM-L6-v2') -> Dict[str, np.ndarray]:
        """Compute sentence embeddings for all text in dataset"""
        if self.sentence_transformer is None:
            logger.info(f"Loading sentence transformer model: {model_name}")
            self.sentence_transformer = SentenceTransformer(model_name)
        
        if not self.examples:
            self.parse_examples()
        
        embeddings = {}
        
        # Collect all texts
        texts = {
            'contexts': [ex.context for ex in self.examples],
            'predictable': [ex.predictable for ex in self.examples],
            'creative': [ex.creative for ex in self.examples],
            'meta_creative': [ex.meta_creative for ex in self.examples]
        }
        
        # Compute embeddings
        for text_type, text_list in texts.items():
            logger.info(f"Computing {text_type} embeddings...")
            embeddings[text_type] = self.sentence_transformer.encode(text_list)
        
        return embeddings
    
    def analyze_surprise_patterns(self) -> pd.DataFrame:
        """Analyze patterns in surprise sources across categories"""
        df = self.get_surprise_analysis_data()
        
        # Count surprise sources by category
        surprise_analysis = df.groupby(['category', 'surprise_source']).size().reset_index(name='count')
        
        # Add percentages
        category_totals = df.groupby('category').size()
        surprise_analysis['percentage'] = surprise_analysis.apply(
            lambda row: (row['count'] / category_totals[row['category']]) * 100, axis=1
        )
        
        return surprise_analysis
    
    def get_examples_by_surprise_source(self, surprise_source: str) -> List[CreativityExample]:
        """Get all examples with a specific surprise source"""
        if not self.examples:
            self.parse_examples()
        
        return [ex for ex in self.examples if ex.surprise_source == surprise_source]
    
    def get_examples_by_category(self, category: str) -> List[CreativityExample]:
        """Get all examples from a specific category"""
        if not self.examples:
            self.parse_examples()
        
        return [ex for ex in self.examples if ex.category == category]
    
    def export_for_training(self, output_dir: str = "training_data"):
        """Export data in formats suitable for different training frameworks"""
        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True)
        
        if not self.examples:
            self.parse_examples()
        
        # Export predictor training data
        predictor_data = self.get_predictor_training_data()
        with open(output_path / "predictor_training.jsonl", 'w') as f:
            for context, predictable in predictor_data:
                f.write(json.dumps({"context": context, "completion": predictable}) + "\n")
        
        # Export generator training data (creative level)
        creative_data = self.get_generator_training_data(creativity_level='creative')
        with open(output_path / "generator_creative.jsonl", 'w') as f:
            for context, creative in creative_data:
                f.write(json.dumps({"context": context, "completion": creative}) + "\n")
        
        # Export generator training data (meta-creative level)
        meta_creative_data = self.get_generator_training_data(creativity_level='meta_creative')
        with open(output_path / "generator_meta_creative.jsonl", 'w') as f:
            for context, meta_creative in meta_creative_data:
                f.write(json.dumps({"context": context, "completion": meta_creative}) + "\n")
        
        # Export analysis data
        df = self.get_surprise_analysis_data()
        df.to_csv(output_path / "full_dataset.csv", index=False)
        
        # Export surprise pattern analysis
        surprise_patterns = self.analyze_surprise_patterns()
        surprise_patterns.to_csv(output_path / "surprise_patterns.csv", index=False)
        
        logger.info(f"Exported training data to {output_path}")
    
    def get_dataset_statistics(self) -> Dict:
        """Get comprehensive dataset statistics"""
        if not self.examples:
            self.parse_examples()
        
        df = self.get_surprise_analysis_data()
        
        stats = {
            'total_examples': len(self.examples),
            'categories': df['category'].value_counts().to_dict(),
            'surprise_sources': df['surprise_source'].value_counts().to_dict(),
            'domains': df['domain'].value_counts().to_dict(),
            'avg_context_length': df['context'].str.len().mean(),
            'avg_predictable_length': df['predictable'].str.len().mean(),
            'avg_creative_length': df['creative'].str.len().mean(),
            'avg_meta_creative_length': df['meta_creative'].str.len().mean()
        }
        
        return stats


def main():
    """Example usage of the dataset loader"""
    
    # Initialize loader with relative path to prompts directory
    loader = CreativityDatasetLoader("../../prompts/creativity_training_dataset_02.json")
    
    # Load and parse dataset
    try:
        dataset = loader.load_dataset()
        examples = loader.parse_examples()
        
        print(f"\n📊 Dataset Statistics:")
        stats = loader.get_dataset_statistics()
        for key, value in stats.items():
            print(f"  {key}: {value}")
        
        print(f"\n🎯 Surprise Source Analysis:")
        surprise_analysis = loader.analyze_surprise_patterns()
        print(surprise_analysis)
        
        # Get training splits
        train_examples, val_examples = loader.get_training_splits()
        
        print(f"\n🔮 Sample Predictor Training Data:")
        predictor_data = loader.get_predictor_training_data(train_examples[:3])
        for i, (context, predictable) in enumerate(predictor_data):
            print(f"  Example {i+1}:")
            print(f"    Context: {context}")
            print(f"    Predictable: {predictable}")
            print()
        
        print(f"🎨 Sample Generator Training Data (Creative):")
        creative_data = loader.get_generator_training_data(train_examples[:3], 'creative')
        for i, (context, creative) in enumerate(creative_data):
            print(f"  Example {i+1}:")
            print(f"    Context: {context}")
            print(f"    Creative: {creative}")
            print()
        
        # Export data for training
        print("📁 Exporting training data...")
        loader.export_for_training()
        
        # Example: Get examples by surprise source
        metaphorical_examples = loader.get_examples_by_surprise_source("unexpected_metaphorical_mapping")
        print(f"\n🎭 Examples with 'unexpected_metaphorical_mapping': {len(metaphorical_examples)}")
        
        # Example: Get examples by category
        visual_examples = loader.get_examples_by_category("visual_descriptions")
        print(f"🖼️ Visual description examples: {len(visual_examples)}")
        
        print("\n✅ Dataset loading and analysis complete!")
        
    except Exception as e:
        logger.error(f"Error loading dataset: {e}")
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())