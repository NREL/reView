# reView

`reView` is a data portal for reviewing Renewable Energy Potential Model ([reV](https://github.com/NREL/reV)) aggregation or supply-curve module outputs. Once a user has generated tables with`reV`, reView allows that user to view the data as an interactive map linked to an interactive graphs, allowing for quick exploration of `reV` outputs. Other functionality allows the user to filter the dataset based on variable thresholds, calculate differences between two tables, compare multiple tables in one chart, group results by region, and calculate least-cost scenarios at each point with more functionality coming soon.

`reView` is currently in under development, but is functional.
<br>

## Installation
1. Clone the `reView` repository.

    Using ssh:
    ```
    git clone git@github.nrel.gov:twillia2/reView.git
    ```
    Using https:
    ```
    git clone https://github.nrel.gov/twillia2/reView.git
    ```

2. Create and activate conda environment:
    1) Create a conda env: ``conda create -n review``
    2) Run the command: ``conda activate review``
    3) cd into the repo: ``cd reView``.
    4) prior to running ``pip`` below, make sure the branch is correct (install from main!)
    5) Install ``reView`` and its dependencies by running:
       ``pip install .`` (or ``pip install -e .[dev]`` if running a dev branch or working on the source code)

3. Create a project config pointing to directory containing reV supply curve tables. Save as json dictionary in configs/ (e.g. configs/sample.json).
```
{
    "project_name": <"Your Project Name>",
    "directory": <"Local path to folder containing reV supply-curve/supply-curve-aggregation module outputs">
}
```
<br>

## Running reView
1. Run the ``reView`` command:
    ```
    reView
    ```
2. Open your browser and enter the url output from command above. The default port is 8050.
    ```
    http://localhost:8050
    ```
<br>

## Running reView with [Gunicorn](https://gunicorn.org/) (Unix only)

1. To run `reView` using `gunicorn` (a Python WSGI HTTP Server for UNIX) for better performance, make sure to follow the installation steps as outlined above, but when you get to the last step of #2, include the `gunicorn` dependency:
    ```
    pip install -e .[gunicorn]
    ```
    or, for more development tools:
    ```
     pip install -e .[dev,gunicorn]
     ```

2. Run `reView` using `gunicorn`:
    ```
    gunicorn reView/index:server
    ```

3. Open your browser and enter the url output from command above. The default port is 9875.
    ```
    http://localhost:9875
    ```
