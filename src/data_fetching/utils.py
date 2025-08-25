"""
Utility functions for data fetching operations.
"""
import json
import uuid
from pathlib import Path
from typing import Any, Dict, List, Optional

from .logging_config import get_logger

logger = get_logger(
    __name__,
    extra={'component': 'DataFetchingUtils'})


def load_json_file(file_path: Path) -> Dict[str, Any]:
    """
    Load and parse a JSON file.

    Args:
        file_path: Path to the JSON file

    Returns:
        Parsed JSON data as a dictionary
    """
    operation_id = str(uuid.uuid4())
    logger.info(
        "Loading JSON file",
        extra={
            'operation': 'json_load',
            'operation_id': operation_id,
            'file_path': str(file_path),
            'file_size': file_path.stat().st_size if file_path.exists() else None
        }
    )

    try:
        with open(file_path, 'r') as file_handle:
            data = json.load(file_handle)
            
        logger.debug(
            "Successfully loaded JSON file",
            extra={
                'operation_id': operation_id,
                'keys': list(data.keys()) if isinstance(data, dict) else None,
                'data_type': type(data).__name__
            }
        )
        return data
        
    except Exception as err:
        logger.error(
            "Failed to load JSON file",
            extra={
                'operation_id': operation_id,
                'error_type': type(err).__name__,
                'error_details': str(err)
            }
        )
        raise


def save_json_file(data: Dict[str, Any], file_path: Path) -> None:
    """
    Save data to a JSON file.

    Args:
        data: The data to save
        file_path: Path where to save the JSON file
    """
    operation_id = str(uuid.uuid4())
    logger.info(
        "Saving JSON file",
        extra={
            'operation': 'json_save',
            'operation_id': operation_id,
            'file_path': str(file_path),
            'data_type': type(data).__name__,
            'data_size': len(str(data))
        }
    )

    try:
        with open(file_path, 'w') as file_handle:
            json.dump(data, file_handle, indent=2)
            
        logger.debug(
            "Successfully saved JSON file",
            extra={
                'operation_id': operation_id,
                'file_size': file_path.stat().st_size,
                'keys': list(data.keys()) if isinstance(data, dict) else None
            }
        )
            
    except Exception as err:
        logger.error(
            "Failed to save JSON file",
            extra={
                'operation_id': operation_id,
                'error_type': type(err).__name__,
                'error_details': str(err)
            }
        )
        raise


def validate_data_pair(data_pair: Dict[str, Any]) -> bool:
    """
    Validate that a data pair has the required structure.

    Args:
        data_pair: The data pair to validate

    Returns:
        True if valid, False otherwise
    """
    operation_id = str(uuid.uuid4())
    logger.debug(
        "Validating data pair",
        extra={
            'operation': 'validate_data',
            'operation_id': operation_id,
            'data_keys': list(data_pair.keys())
        }
    )

    required_fields = ['input', 'output']
    is_valid = all(field in data_pair for field in required_fields)

    if not is_valid:
        logger.warning(
            "Data pair validation failed",
            extra={
                'operation_id': operation_id,
                'missing_fields': [field for field in required_fields if field not in data_pair]
            }
        )
    else:
        logger.debug(
            "Data pair validation successful",
            extra={
                'operation_id': operation_id,
                'input_type': type(data_pair['input']).__name__,
                'output_type': type(data_pair['output']).__name__
            }
        )

    return is_valid
