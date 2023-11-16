# Email Classification using Logistic Regression

This repository contains code for a simple email classification project using Logistic Regression. The model is trained to classify emails into spam or non-spam (ham) categories. Follow the steps below to set up and run the project.

## Getting Started

### Prerequisites

Make sure you have Python and pip installed on your system.

### Setting up a Virtual Environment

```bash
# Activate the virtual environment
# On Windows
venv\Scripts\activate
# On macOS and Linux
source venv/bin/activate
```

### Install Dependencies

Install the required packages listed in the `requirements.txt` file:

```bash
pip install -r requirements.txt
```

### Dataset

Download the dataset (`mail_data.csv`) and place it in the project directory.

## Running the Code

Navigate to the project directory and run the script:

```bash
python email_classification.py
```

This script reads the data, preprocesses it, trains a logistic regression model, and evaluates its accuracy on training and testing sets. Additionally, you can add your own email to check whether it is classified as spam or not.

```python
# Add your own mail here & run the code to see if it is a spam mail or not!
input_your_mail = ['Your email content goes here']  # Please enter your emails here
input_data_features = feature_extraction.transform(input_your_mail)

prediction = model.predict(input_data_features)

if prediction[0] == 1:
    print('This mail is not a spam mail')
else:
    print('This mail is a spam mail')
```

## Deactivating the Virtual Environment

When you're done, deactivate the virtual environment:

```bash
# On Windows
deactivate
# On macOS and Linux
deactivate
```

## Contributing

If you'd like to contribute to this project, fork the repository and submit a pull request. Contributions are welcome!

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- Special thanks to the scikit-learn and pandas libraries for their contributions to this project.