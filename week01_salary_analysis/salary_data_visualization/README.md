📊 Salary Analysis in Iran
This project uses Python's data analysis and visualization libraries to explore and visualize salary data from various jobs across different cities in Iran. The dataset is expected to be in a CSV format (salaries_iran (1).csv), containing at least the following columns:

salary: Salary in numeric or string format (with commas)

job_title: Job title or position

city: City where the job is located

experience_years: Years of experience

📦 Requirements
Python 3.x

pandas

seaborn

matplotlib

You can install the necessary libraries using:

bash
Copy
Edit
pip install pandas seaborn matplotlib
📁 Files
salaries_iran (1).csv: Input dataset containing salary information.

salary_analysis.py: Python script for data processing and visualization.

🔍 What the Script Does
Read and Clean the Data

Loads salary data from the CSV file.

Cleans the salary column by removing commas and converting it to integers.

Salary Distribution Plot

Uses a histogram to visualize the distribution of salaries across all entries.

Average Salary per Job Title

Groups the data by job title and calculates the average salary.

Displays the result using a bar plot (with the "viridis" color palette).

Average Salary per City

Groups the data by city and calculates the average salary.

Displays the result using a bar plot (with the "magma" color palette).

Salary vs Experience Scatter Plot

Creates a scatter plot to visualize the relationship between salary and years of experience.

Correlation Heatmap

Computes correlation between numeric columns (e.g., salary, experience).

Displays a heatmap with annotation for easy interpretation.

🖼️ Output
The script generates the following visualizations:

Histogram of salary distribution

Bar plot of average salary per job

Bar plot of average salary per city

Scatter plot of salary vs experience

Correlation matrix heatmap

📌 Notes
Make sure the CSV file is in the same directory as the script, or provide the correct path.

The script assumes that the salary column contains strings with comma separators (e.g., "1,200,000"), which it cleans before analysis.