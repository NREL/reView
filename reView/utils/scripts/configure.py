# -*- coding: utf-8 -*-
"""Configure project from reV configs.

Created on Wed Jul 14 12:13:37 2021

@author: twillia2
"""
import json
from pathlib import Path

from utils.config import Config
from old.revlogs import find_files


PROJECT = "ATB Onshore - FY21"


class Configure(Config):
    """Methods for inferring needed information from the reV configs."""

    def __init__(
        self, project=None, pwd=".", pipe_name="config_pipeline.json"
    ):  # pylint: disable=redefined-outer-name
        """Initialize Configure object."""
        super().__init__(project)
        self.pwd = Path(pwd)
        self.pipe_name = pipe_name

    def __repr__(self):  # pylint: disable=redefined-outer-name
        """Print representation string."""
        args = ", ".join([f"{k}={v}" for k, v in self.__dict__.items()])
        msg = f"<reView Configure object: {args}>"
        return msg

    def from_pipeline(self, pipe_path):  # pylint: disable=redefined-outer-name
        """Use the pipeline to locate all needed information for a run."""
        with open(pipe_path, "r") as file:
            config = json.load(file)  # pylint: disable=unused-variable

    def from_pipelines(self):  # pylint: disable=redefined-outer-name
        """Use a folder containing multiple pipelines to find information."""
        # Find all the pipeline files
        pipes = find_files(self.pwd, self.pipe_name)
        pipes.sort()


if __name__ == "__main__":
    self = Configure(PROJECT)
