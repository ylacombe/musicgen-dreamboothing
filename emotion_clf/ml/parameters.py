"""Parameters class"""

import json
import logging
import os
from typing import Any, Dict, Iterable, Optional

class Parameters():
    """Contains parameters"""
    def save(self, json_path: str) -> None:
        """Save parameters to json file"""
        with open(json_path, 'w', encoding="utf-8") as file:
            json.dump(self.__dict__, file, indent=4)

    def update(self, params_dict: Dict[str, Any]) -> None:
        """Update parameters from Dict"""
        self.__dict__.update(params_dict)

    def update_from_file(self, json_path: str) -> None:
        """Update parameters from file"""
        with open(json_path, encoding="utf-8") as file:
            params = json.load(file)
            self.update(params)

    def show(self) -> str:
        """Print parameters to string for display"""
        return json.dumps(self.__dict__, indent=2)

    def get(self, *args: str) -> Any:
        """Get nested value. Returns None if one of the keys is absent"""
        return get_value(self.__dict__, args)

    def set(self, value, *args: str):
        """Set nested value"""
        set_value(self.__dict__, value, args)


def set_value(dictionnary: Dict[str, Any], value: Any,
              key_list: Iterable[str]):
    """Set nested value in Dict"""
    head, *tail = key_list
    if len(tail) == 0:
        dictionnary[head] = value
    else:
        if head not in dictionnary:
            dictionnary[head] = {}
        set_value(dictionnary[head], value, tail)


def get_value(dictionnary: Optional[Dict[str, Any]],
              key_list: Iterable[str]) -> Any:
    """Get nested value from Dict. Returns None if one of the keys is absent"""
    if dictionnary is None:
        result = None
    else:
        head, *tail = key_list
        if len(tail) == 0:
            result = dictionnary.get(head)
        else:
            result = get_value(dictionnary.get(head), tail)
    return result


def load_parameters_recursively(directory: str) -> Parameters:
    """Load Parameters from file

    Load parameters recursively from all parameters.json files. Child directory
    overrides parents for each first level key."""
    params = Parameters()
    json_path = os.path.join(directory, "parameters.json")
    if os.path.isfile(json_path):
        #logging.info("Loading Parameters from %s", json_path)
        params.update_from_file(json_path)
    return params
