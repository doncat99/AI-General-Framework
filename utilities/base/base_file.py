# utilities/base/base_file.py
from typing import Any, Optional, Type, TypeVar, Union
import os
import json
from pydantic import BaseModel
from loguru import logger

T = TypeVar('T', bound=BaseModel)


class BaseFile():
    """Base class for agents with file handling and caching capabilities."""

    def read_json_file(self, file_path: str, model: Optional[Type[T]] = None) -> Union[T, Any, None]:
        """
        Read a JSON file and optionally validate it with a Pydantic model.

        Args:
            file_path: Path to the JSON file.
            model: Optional Pydantic model to validate the JSON content.

        Returns:
            Parsed JSON content (as a Pydantic model instance or raw dictionary/list), or None if reading fails.
        """
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            if model:
                return model.model_validate(data)
            return data
        except FileNotFoundError:
            # logger.warning(f"File not found: {file_path}")
            return None
        except json.JSONDecodeError as e:
            logger.error(f"JSON decoding error in {file_path}: {str(e)}")
            return None
        except Exception as e:
            logger.error(f"Failed to read {file_path}: {str(e)}")
            return None

    def write_json_file(self, file_path: str, data: Any, indent: int = 2) -> bool:
        """
        Write data to a JSON file.

        Args:
            file_path: Path to the JSON file.
            data: Data to write (e.g., dictionary, list, or Pydantic model).
            indent: Indentation level for JSON formatting (default: 2).

        Returns:
            True if writing succeeds, False otherwise.
        """
        try:
            if isinstance(data, BaseModel):
                data = data.model_dump()
            elif isinstance(data, list) and all(isinstance(item, BaseModel) for item in data):
                data = [item.model_dump() for item in data]
            with open(file_path, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=indent)
            return True
        except Exception as e:
            logger.error(f"Failed to write to {file_path}: {str(e)}")
            return False

    def delete_file(self, file_path: str) -> bool:
        """
        Delete a file if it exists.

        Args:
            file_path: Path to the file to delete.

        Returns:
            True if deletion succeeds or file doesn't exist, False otherwise.
        """
        try:
            if os.path.exists(file_path):
                os.remove(file_path)
                logger.debug(f"Deleted file: {file_path}")
            return True
        except Exception as e:
            logger.error(f"Failed to delete {file_path}: {str(e)}")
            return False