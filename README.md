## Smart Inventory Manager: A Web-Based Inventory Management System with Computer Vision
This project utilizes Long Short-Term Memory (LSTM) neural networks to predict future sales of different product models based on historical stock data. It takes two JSON files as input: one containing the initial stock (initial_stock.json) and another tracking daily sales (sold_stock.json). The LSTM model analyzes past sales trends to forecast the quantity of each product expected to be sold over the next 90 days. The predictions are visualized using bar charts in a Streamlit web application, providing valuable insights for inventory management and demand forecasting, helping businesses optimize stock levels efficiently.

## About
<!--Detailed Description about the project-->
This project focuses on sales prediction using Long Short-Term Memory (LSTM) neural networks to help businesses forecast future demand and optimize inventory management. The system takes two JSON files as input: initial_stock.json, which contains the starting inventory for various product models, and sold_stock.json, which logs daily sales transactions for each model over a specific period. The data is preprocessed to structure it into time-series format, ensuring that missing dates are accounted for and sales data is properly aggregated. The LSTM model is then trained to recognize historical sales patterns and predict the expected sales quantities for the next 90 days. To improve efficiency, the trained models are saved and reloaded for future use instead of being retrained each time. The predictions are displayed using bar charts in a user-friendly Streamlit web application, allowing users to visually interpret the forecasted sales trends for different product categories. This system provides valuable insights for inventory planning, helping businesses make data-driven decisions to prevent overstocking or understocking of products. By leveraging machine learning, the project enhances the accuracy of demand forecasting, ultimately improving operational efficiency and profitability.

## Features
<!--List the features of the project as shown below-->
- Implements a LSTM Model.
- A framework based application for deployment purpose.
- High scalability.
- Less time complexity.


## Requirements
<!--List the requirements of the project as shown below-->
* Operating System: Requires a 64-bit OS (Windows 10 or Ubuntu) for compatibility with deep learning frameworks.
* Development Environment: Python 3.6 or later is necessary for coding the sign language detection system.
* Deep Learning Frameworks: TensorFlow for model training, MediaPipe for hand gesture recognition.

* Version Control: Implementation of Git for collaborative development and effective code management.
* IDE: Use of Jupyter Notebooks as the Integrated Development Environment for coding, debugging, and version control integration.
* Additional Dependencies: Includes scikit-learn, TensorFlow (versions 2.4.1), TensorFlow GPU, OpenCV, and Mediapipe for deep learning tasks.

## System Architecture
<!--Embed the system architecture diagram as shown below-->

![architecture-diagram](https://github.com/user-attachments/assets/6197f1e3-21b9-4412-9ca7-e04e7e32fd4b)

## Output

<!--Embed the Output picture at respective places as shown below as shown below-->
#### Output1 - Name of the output

![output-01](https://github.com/user-attachments/assets/552a189f-4b93-4f4e-9fd5-6c6e13f6fd77)

#### Output2 - Name of the output

![output-02](https://github.com/user-attachments/assets/1b5742bb-46d1-4974-b0da-82ff6139c27d)

Detection Accuracy: 96.7%
Note: These metrics can be customized based on your actual performance evaluations.


## Results and Impact
<!--Give the results and impact as shown below-->
The project successfully predicts future sales trends for different product models using LSTM-based time series forecasting. The trained models analyze historical sales data and generate predictions for the next 90 days, which are visualized using bar charts in the Streamlit web application. The results show that the LSTM model effectively captures seasonal patterns and demand fluctuations, helping businesses anticipate stock requirements. The prediction accuracy depends on the quality and consistency of past sales data, with better results observed for products with stable sales histories. The inventory optimization insights enable businesses to make informed restocking decisions, preventing both overstocking and stockouts. However, challenges such as data sparsity, missing values, and sudden demand shifts can impact prediction accuracy, requiring further improvements like hyperparameter tuning, additional data preprocessing, or hybrid models. Overall, the project provides a reliable and automated sales forecasting solution, improving decision-making and profitability for businesses managing inventory. 🚀📊

## Articles published / References
1. S. Singh, R. Kumar, A. Badhoutiya, U. Sharma, A. Alkhayyat and S. K. Shah, "Web-Based Inventory, Stock Monitoring and Control System Powered by Local Encrypted Web Server," 2024 11th International Conference on Computing for Sustainable Global Development (INDIACom), New Delhi, India, 2024, pp. 741-744, doi: 10.23919/INDIACom61295.2024.10498795
2. T. Balaji, V. Hari, S. Lathifunnisa, P. Ganesh and P. Arupya, "Optimizing Web-based Inventory Management system using QR code Technology," 2024 7th International Conference on Circuit Power and Computing Technologies (ICCPCT), Kollam, India, 2024, pp. 751-756, doi: 10.1109/ICCPCT61902.2024.10673103
3. J. P. N, S. Prashanth, A. D and M. J, "Design and Evaluation of a Real-Time Stock Inventory Management System," 2023 IEEE 5th International Conference on Cybernetics, Cognition and Machine Learning Applications (ICCCMLA), Hamburg, Germany, 2023, pp. 180-185, doi: 10.1109/ICCCMLA58983.2023.10346665
4. N. K. Verma, T. Sharma, S. D. Rajurkar and A. Salour, "Object identification for inventory management using convolutional neural network," 2016 IEEE Applied Imagery Pattern Recognition Workshop (AIPR), Washington, DC, USA, 2016, pp. 1-6, doi: 10.1109/AIPR.2016.8010578




