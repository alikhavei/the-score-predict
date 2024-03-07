import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
lst_s_pass = []
lst_p_pass = []
lst_s_fail = []
lst_p_fail = []
x = pd.read_csv('student_exam_data.csv')
df = pd.DataFrame(x)

pass_or_fail = df['Pass/Fail']

passes = df[pass_or_fail == 1]
failed = df[pass_or_fail == 0]

Study = passes['Study Hours']
previous = passes['Previous Exam Score']
# #----------------------------------------------
Study_fail = failed['Study Hours']
previous_fail = failed['Previous Exam Score']
index_passes =list(passes.index)
index_failed = list(failed.index)
for i in index_passes:
    lst_s_pass.append(Study[i])
    lst_p_pass.append(previous[i])
for r in index_failed:
    lst_s_fail.append(Study_fail[r])
    lst_p_fail.append(previous_fail[r])

lst_s_pass = np.array(lst_s_pass)
lst_p_pass = np.array(lst_p_pass)
plt.scatter(lst_s_pass,lst_p_pass,c='blue')
#---------------------------------
lst_s_fail = np.array(lst_s_fail)
lst_p_fail = np.array(lst_p_fail)
plt.scatter(lst_s_fail,lst_p_fail,c="red")
plt.xlabel('Study hours')
plt.ylabel('Previos exam score')

plt.show()