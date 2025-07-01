import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt


df=pd.read_csv("salaries_iran (1).csv")
df['salary'] = df['salary'].str.replace(',', '').astype(int)
sns.histplot(df['salary'])
plt.title("Salary Distribution")
plt.xlabel("Salary")
plt.ylabel("Count")
plt.show()

avg_salary_per_job=df.groupby('job_title')['salary'].mean().sort_values(ascending=False)
sns.barplot(x=avg_salary_per_job.index,y=avg_salary_per_job.values,palette="viridis")
plt.title("Avg_salary per job")
plt.xlabel("Job")
plt.ylabel("Avg_salary")
plt.show()



avg_salary_per_city=df.groupby('city')['salary'].mean().sort_values(ascending=False)
sns.barplot(x=avg_salary_per_city.index,y=avg_salary_per_city.values,palette="magma")
plt.title("Avg_salary per city")
plt.xlabel("City")
plt.ylabel("Avg_salary")
plt.show()

sns.scatterplot(x=df['salary'],y=df['experience_years'])
plt.title("Salary VS Experience_years")
plt.xlabel("Salary")
plt.ylabel("Experience")
plt.show()


numeric_df = df.select_dtypes(include=['number'])
cor = numeric_df.corr()
sns.heatmap(cor,annot=True,cmap='coolwarm')
plt.title('Correlation Matrix')
plt.show()