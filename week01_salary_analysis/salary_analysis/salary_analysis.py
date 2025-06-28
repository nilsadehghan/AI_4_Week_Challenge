import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
df=pd.read_csv("salaries_iran (1).csv")
print(df.head(5))
df["salary"]=df["salary"].str.replace(',','').astype(int)
mean_salary=np.mean(df['salary'])
print("mean_salary:",mean_salary)
median_salary=np.median(df['salary'])
print("median_salary:",median_salary)
avg_salary_job=df.groupby('job_title')['salary'].mean()
top_job=avg_salary_job.idxmax()
top_salary=avg_salary_job.max()
print("top_job:",top_job)
print('top_salary:',top_salary)

avg_salary_city=df.groupby('city')['salary'].mean()
print("avg_salary_city:",avg_salary_city)


plt.bar(avg_salary_job.index,avg_salary_job.values,color="pink")
plt.xlabel("job")
plt.ylabel("salary")
# plt.xticks(rotation=45)
plt.show()
plt.scatter(df['experience_years'],df['salary'],color="blue")
plt.xlabel("experience_years")
plt.ylabel("salary")
plt.show()
corr_matrix=np.corrcoef(df['experience_years'],df['salary'])
corr=corr_matrix[0,1]
print("corr:",corr)