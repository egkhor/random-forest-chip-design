# Random Forest for Chip Design Defect Prediction

This open-source project applies the **Random Forest** algorithm to predict defects in semiconductor chip designs, achieving ~85% accuracy on a synthetic dataset of 1,000 samples. It analyzes structured data (e.g., transistor count, defect rate) to optimize chip design workflows.

## Features
- **Random Forest Model**: Ensemble of Decision Trees for robust defect classification.
- **Synthetic Dataset**: 1,000 samples for prototyping.
- **Scalable**: Built with scikit-learn for local execution.

## Algorithm Comparison
- **Random Forest**: Aggregates trees for high accuracy, handles non-linear data, interpretable via feature importance.
- **Decision Trees**: Simple, interpretable, but prone to overfitting.
- **SVMs**: Effective for non-linear data, less interpretable, needs feature scaling.
- **LLMs**: Suited for text (e.g., design notes), compute-heavy, less effective for tabular data.

## Project Structure
```
random-forest-chip-design/
├── README.md
├── requirements.txt
├── .gitignore
├── CONTRIBUTING.md
├── LICENSE
├── src/
│   └── random_forest_defect_prediction.py
└── data/
    ├── chip_defect_data.csv
    └── random_forest_metrics.txt
```

## Getting Started
### Prerequisites
- Python 3.8+
- Dependencies: `pip install -r requirements.txt`

### Installation
1. Clone the repository:
   ```bash
   git clone https://github.com/egkhor/random-forest-chip-design.git
   cd random-forest-chip-design
   ```
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Run the script:
   ```bash
   python src/random_forest_defect_prediction.py
   ```

### Output
- Generates `chip_defect_data.csv` and `random_forest_metrics.txt` with model accuracy (~85%).

## Contributing
See [CONTRIBUTING.md](CONTRIBUTING.md) to add datasets, models, or visualizations.

## License
MIT License. See [LICENSE](LICENSE).

## Contact
Open an Issue or join Discussions on [GitHub](https://github.com/egkhor/random-forest-chip-design).