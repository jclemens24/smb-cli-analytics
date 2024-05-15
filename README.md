# Command Syntax Differences Report Generator

This project generates a report of command syntax differences across different products.

Ensure to have Python 3.6 or later installed on your machine.

## Installation

1. Clone this repository:

   ```bash
   git clone https://github.com/jclemens24/smb-cli-analytics.git
   ```

2. Navigate to the project directory:

   ```bash
   cd smb-cli-analytics
   ```

3. Install the required Python packages:

   ```bash
   pip install -r requirements.txt
   ```

## Usage

Run the script with the following command:

```bash
python support_docs_loader.py
```

The script will generate a CSV report named `command_syntax_diffs_with_products.csv` in the `data/reports` directory.
There will be a number of other reports generated in the `data/reports` directory as well.
Add product urls as needed for comparison within `support_docs_loader.py`.

## Contributing

Pull requests are welcome. For major changes, please open an issue first to discuss what you would like to change.

## License

[MIT](https://choosealicense.com/licenses/mit/)
