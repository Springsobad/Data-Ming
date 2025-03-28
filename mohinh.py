import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from scipy.stats import chi2_contingency
from sklearn.metrics import precision_score, recall_score, f1_score
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB, MultinomialNB
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import LabelEncoder
import pickle
from tkinter import Tk
from tkinter.filedialog import askopenfilename
from sklearn.metrics import confusion_matrix
# Tải tệp từ máy tính của bạn
Tk().withdraw()  # Không hiển thị cửa sổ chính
file_path = askopenfilename(title="Select a file", filetypes=[("CSV files", "*.csv")])  # Mở hộp thoại chọn tệp

# Đọc dữ liệu từ tệp CSV
df = pd.read_csv(file_path)

# In các thông tin cơ bản về dữ liệu
column_names = df.columns.tolist()
print("Columns:", column_names)
print("Number of columns:", len(column_names))

# Kiểm tra giá trị thiếu
empty_values_count = df.isnull().sum()
print("Empty Values:", empty_values_count)

# Kiểu dữ liệu của các cột
column_data_types = df.dtypes
print("Data Types:", column_data_types)

# Số lượng dòng và cột
num_rows = len(df)
num_columns = len(df.columns)
print("\nData Information:")
print(f"Number of rows: {num_rows}")
print(f"Number of columns: {num_columns}")

# Kiểm tra các giá trị khác nhau trong các cột dạng object
object_columns = df.select_dtypes(include='object').columns
distinct_values = {}

for column in object_columns:
    distinct_values[column] = df[column].unique()

# In các giá trị khác nhau
for column, values in distinct_values.items():
    print(f"Column '{column}':")
    print(values)

# Chi-square test để đánh giá các biến phân loại
categorical_vars = ['cap-shape', 'cap-surface', 'cap-color', 'bruises', 'odor', 'gill-attachment',
                    'gill-spacing', 'gill-size', 'gill-color', 'stalk-shape', 'stalk-root',
                    'stalk-surface-above-ring', 'stalk-surface-below-ring', 'stalk-color-above-ring',
                    'stalk-color-below-ring', 'veil-type', 'veil-color', 'ring-number', 'ring-type',
                    'spore-print-color', 'population', 'habitat']
target_var = 'class'

# Lưu kết quả chi-square
chi2_results = []

for var in categorical_vars:
    contingency_table = pd.crosstab(df[var], df[target_var])
    chi2, p_value, dof, expected = chi2_contingency(contingency_table)
    chi2_results.append((var, chi2, p_value))

# Sắp xếp kết quả theo p-value
chi2_results.sort(key=lambda x: x[2])
variables = []
chi2_values = []
p_values = []

# Lưu kết quả vào danh sách
for var, chi2, p_value in chi2_results:
    variables.append(var)
    chi2_values.append(chi2)
    p_values.append(p_value)

# Vẽ biểu đồ chi-square
plt.figure(figsize=(10, 8))
plt.barh(variables, chi2_values, color='steelblue')
plt.xlabel('Chi-square value')
plt.ylabel('Categorical Variable')
plt.title('Importance of Categorical Variables')
plt.xscale('log')
plt.grid(axis='x', linestyle='--')
plt.tight_layout()

# Thêm giá trị p vào biểu đồ
for i, value in enumerate(p_values):
    plt.annotate(f'p={value:.2e}', (chi2_values[i], i), xytext=(5, 8), textcoords='offset points', fontsize=9)

plt.show()

# Tiền xử lý dữ liệu: chọn một số cột và mã hóa chúng
newDf = df[["class",'odor', 'spore-print-color', 'gill-color', 'stalk-surface-above-ring', 'stalk-surface-below-ring', 'ring-type']]

# Vẽ các biểu đồ phân phối theo từng cột
columns = ['odor', 'spore-print-color', 'gill-color', 'stalk-surface-above-ring', 'stalk-surface-below-ring', 'ring-type']
plt.figure(figsize=(12, 8))

for i, column in enumerate(columns, start=1):
    plt.subplot(2, 3, i)
    sns.set(style="darkgrid")
    colors = sns.color_palette('Set3', len(newDf[column].unique()))
    newDf[column].value_counts().plot(kind='pie', autopct='%1.1f%%', colors=colors)
    plt.title(column, fontsize=14, fontweight='bold')
    plt.axis('equal')

plt.tight_layout()
plt.show()

# Đổi thành đồ thị đếm theo 'class'
plt.figure(figsize=(12, 8))

for i, column in enumerate(columns, start=1):
    plt.subplot(2, 3, i)
    sns.countplot(x=column, hue='class', data=newDf)
    plt.title(column, fontsize=14, fontweight='bold')

plt.tight_layout()
plt.show()

# Hàm mã hóa nhãn
def label_encode_categorical(df):
    encoded_df = pd.DataFrame()
    encoding_mapping = {}

    for column in df.columns:
        if df[column].dtype == object:
            label_encoder = LabelEncoder()
            encoded_column = label_encoder.fit_transform(df[column])
            encoded_df[column] = encoded_column
            encoding_mapping[column] = dict(zip(label_encoder.classes_, label_encoder.transform(label_encoder.classes_)))
        else:
            encoded_df[column] = df[column]

    return encoded_df, df, encoding_mapping

encoded_df, original_df , encoding_mapping = label_encode_categorical(newDf)
print(encoding_mapping)

y = encoded_df[["class"]]
x = encoded_df.drop(['class'], axis=1)

# Chia dữ liệu thành các tập train, validation, test
X_train, X_temp, y_train, y_temp = train_test_split(x, y, test_size=0.3, random_state=42)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.17, random_state=42)

print("X_train shape:", X_train.shape, "y_train shape:", y_train.shape)
print("X_val shape:", X_val.shape, "y_val shape:", y_val.shape)
print("X_test shape:", X_test.shape, "y_test shape:", y_test.shape)

# Chuyển y thành mảng 1D
y_train = y_train.values.ravel()
y_val = y_val.values.ravel()
y_test = y_test.values.ravel()

# Chuyển X thành mảng NumPy nếu không cần tên cột
X_train = X_train.values
X_val = X_val.values
X_test = X_test.values

def plot_confusion_matrix(cm, model_name):
    plt.figure(figsize=(6, 4))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.xlabel('Predicted Labels')
    plt.ylabel('True Labels')
    plt.title(f'{model_name} - Confusion Matrix')
    plt.show()

# Các mô hình
def run_random_forest(X_train, X_val, X_test, y_train, y_val, y_test):
    model = RandomForestClassifier()
    model.fit(np.concatenate((X_train, X_val)), np.concatenate((y_train, y_val)))
    y_pred_test = model.predict(X_test)
    cm = confusion_matrix(y_test, y_pred_test)
    plot_confusion_matrix(cm, 'Random Forest')
    results = {
        "model": model,
        "Test Accuracy": accuracy_score(y_test, y_pred_test),
        "Precision": precision_score(y_test, y_pred_test, average='weighted'),
        "Recall": recall_score(y_test, y_pred_test, average='weighted'),
        "F1 Score": f1_score(y_test, y_pred_test, average='weighted')
    }
    return results

def run_logistic_regression(X_train, X_val, X_test, y_train, y_val, y_test):
    model = LogisticRegression(max_iter=1000)
    model.fit(np.concatenate((X_train, X_val)), np.concatenate((y_train, y_val)))
    y_pred_test = model.predict(X_test)
    cm = confusion_matrix(y_test, y_pred_test)
    plot_confusion_matrix(cm, 'Logistic Regression')
    results = {
        "model": model,
        "Test Accuracy": accuracy_score(y_test, y_pred_test),
        "Precision": precision_score(y_test, y_pred_test, average='weighted'),
        "Recall": recall_score(y_test, y_pred_test, average='weighted'),
        "F1 Score": f1_score(y_test, y_pred_test, average='weighted')
    }
    return results

def run_naive_bayes(X_train, X_val, X_test, y_train, y_val, y_test):
    model = MultinomialNB()
    model.fit(np.concatenate((X_train, X_val)), np.concatenate((y_train, y_val)))
    y_pred_test = model.predict(X_test)
    cm = confusion_matrix(y_test, y_pred_test)
    plot_confusion_matrix(cm, 'Naive Bayes')
    results = {
        "model": model,
        "Test Accuracy": accuracy_score(y_test, y_pred_test),
        "Precision": precision_score(y_test, y_pred_test, average='weighted'),
        "Recall": recall_score(y_test, y_pred_test, average='weighted'),
        "F1 Score": f1_score(y_test, y_pred_test, average='weighted')
    }
    return results

def run_decision_tree(X_train, X_val, X_test, y_train, y_val, y_test):
    model = DecisionTreeClassifier()
    model.fit(np.concatenate((X_train, X_val)), np.concatenate((y_train, y_val)))
    y_pred_test = model.predict(X_test)
    cm = confusion_matrix(y_test, y_pred_test)
    plot_confusion_matrix(cm, 'Decision Tree')
    results = {
        "model": model,
        "Test Accuracy": accuracy_score(y_test, y_pred_test),
        "Precision": precision_score(y_test, y_pred_test, average='weighted'),
        "Recall": recall_score(y_test, y_pred_test, average='weighted'),
        "F1 Score": f1_score(y_test, y_pred_test, average='weighted')
    }
    return results

def run_gradient_boosting(X_train, X_val, X_test, y_train, y_val, y_test):
    model = GradientBoostingClassifier()
    model.fit(np.concatenate((X_train, X_val)), np.concatenate((y_train, y_val)))
    y_pred_test = model.predict(X_test)
    cm = confusion_matrix(y_test, y_pred_test)
    plot_confusion_matrix(cm, 'Gradient Boosting')
    results = {
        "model": model,
        "Test Accuracy": accuracy_score(y_test, y_pred_test),
        "Precision": precision_score(y_test, y_pred_test, average='weighted'),
        "Recall": recall_score(y_test, y_pred_test, average='weighted'),
        "F1 Score": f1_score(y_test, y_pred_test, average='weighted')
    }
    return results
# Chạy tất cả các mô hình và lấy kết quả cross-validation accuracy
results_nb = run_naive_bayes(X_train, X_val, X_test, y_train, y_val, y_test)
results_dt = run_decision_tree(X_train, X_val, X_test, y_train, y_val, y_test)
results_gb = run_gradient_boosting(X_train, X_val, X_test, y_train, y_val, y_test)

# Danh sách các mô hình và độ chính xác cross-validation
models = [
    "Naive Bayes",
    "Decision Tree",
    "Gradient Boosting"
]

cv_accuracies = [
    results_nb["Test Accuracy"],  # Độ chính xác từ mô hình Naive Bayes
    results_dt["Test Accuracy"],  # Độ chính xác từ mô hình Decision Tree
    results_gb["Test Accuracy"]   # Độ chính xác từ mô hình Gradient Boosting
]

# Vẽ biểu đồ so sánh
x = np.arange(len(models))  # vị trí cho từng mô hình
width = 0.5  # độ rộng của mỗi cột

fig, ax = plt.subplots(figsize=(8, 6))

# Vẽ cột cho Cross-Validation Accuracy
bars = ax.bar(x, cv_accuracies, width, label='Test Accuracy', color='blue')

# Định dạng biểu đồ
ax.set_xlabel("Models")
ax.set_ylabel("Accuracy")
# ax.set_title("Bảng so sánh Độ chính xác của các mô hình")
ax.set_xticks(x)
ax.set_xticklabels(models)
ax.legend()

# Hiển thị giá trị trên các cột
for bar in bars:
    yval = bar.get_height()
    ax.text(bar.get_x() + bar.get_width()/2, yval + 0.01, round(yval, 3), ha='center', va='bottom')

plt.ylim(0, 1)  # Giới hạn trục y từ 0 đến 1
plt.show()


# Chạy tất cả các mô hình
results_rf = run_random_forest(X_train, X_val, X_test, y_train, y_val, y_test)
results_lr = run_logistic_regression(X_train, X_val, X_test, y_train, y_val, y_test)
results_nb = run_naive_bayes(X_train, X_val, X_test, y_train, y_val, y_test)
results_dt = run_decision_tree(X_train, X_val, X_test, y_train, y_val, y_test)
results_gb = run_gradient_boosting(X_train, X_val, X_test, y_train, y_val, y_test)

# Lưu mô hình vào file pkl
model_files = {
    "Random Forest": "random_forest.pkl",
    "Logistic Regression": "logistic_regression.pkl",
    "Naive Bayes": "naive_bayes.pkl",
    "Decision Tree": "decision_tree.pkl",
    "Gradient Boosting": "gradient_boosting.pkl"
}

model_results = {
    "Random Forest": results_rf['model'],
    "Logistic Regression": results_lr['model'],
    "Naive Bayes": results_nb['model'],
    "Decision Tree": results_dt['model'],
    "Gradient Boosting": results_gb['model']
}

# Lưu mô hình vào file pkl
for model_name, filename in model_files.items():
    with open(filename, "wb") as f:
        pickle.dump(model_results[model_name], f)
