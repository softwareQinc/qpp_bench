# Benchmarking suite

To run, follow the steps below.

- Create a virtual environment and activate it

```shell
python3 -mvenv venv --prompt bench
source venv/bin/activate
```

On Windows platforms, replace the activation line with either
`bench\Scripts\activate.bat` (Command Prompt) or
`.\bench\Scripts\Activate.ps1` (PowerShell).

- Install the requirements

```shell
pip install -r requirements.txt
```

- Run the benchmark

```shell
python bench.py
```

To run the native C++ `QPPEngine`, first build the required executables

```shell
cmake -S cpp -B cpp/build
cmake --build cpp/build
```

- For additional help, run

```shell
python bench.py --help
```
