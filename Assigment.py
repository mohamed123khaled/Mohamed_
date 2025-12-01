#1

# import pandas as pd
# from sklearn.datasets import load_breast_cancer
# from sklearn.preprocessing import StandardScaler
# from sklearn.linear_model import LogisticRegression
# from sklearn.model_selection import train_test_split
# import matplotlib.pyplot as plt
# data = load_breast_cancer()
# X = pd.DataFrame(data.data, columns=data.feature_names)
# y = pd.Series(data.target)
# X_train, X_test, y_train, y_test = train_test_split(
#     X, y, test_size=0.2, random_state=42
# )
# scaler = StandardScaler()
# X_train_scaled = scaler.fit_transform(X_train)

# model = LogisticRegression(max_iter=500)
# model.fit(X_train_scaled, y_train)
# importance = pd.Series(
#     model.coef_[0],
#     index=X.columns
# ).sort_values(ascending=False)

# print("Top Important Features:\n")
# print(importance.head(10))
# plt.figure(figsize=(10, 8))
# importance.head(15).plot(kind='barh')
# plt.title("Top 15 Important Features")
# plt.xlabel("Coefficient Value")
# plt.gca().invert_yaxis()
# plt.show()















#2 

# import pandas as pd
# from sklearn.datasets import load_breast_cancer
# from sklearn.model_selection import train_test_split
# from sklearn.preprocessing import StandardScaler
# from sklearn.linear_model import LogisticRegression
# from sklearn.pipeline import Pipeline
# from sklearn.metrics import accuracy_score
# data = load_breast_cancer()
# X = pd.DataFrame(data.data, columns=data.feature_names)
# y = pd.Series(data.target)
# X_train, X_test, y_train, y_test = train_test_split(
#     X, y, test_size=0.2, random_state=42
# )
# pipeline = Pipeline([
#     ('scaler', StandardScaler()),       
#     ('model', LogisticRegression(max_iter=300))
# ])
# pipeline.fit(X_train, y_train)
# y_pred = pipeline.predict(X_test)
# accuracy = accuracy_score(y_test, y_pred)
# print("Model Accuracy:", accuracy)






 