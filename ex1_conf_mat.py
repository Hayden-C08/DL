import seaborn as sns
import pandas as pd
from pycm import ConfusionMatrix
import matplotlib.pyplot as plt

confusion_matrix = {
    "Red Disease": {"Red Disease": 92, "Aeromoniasis": 0, "Gill Disease": 3, "Saprolegniasis": 2, "Healthy Fish": 0, "Parasitic Disease": 0, "White Tail Disease": 3},
    "Aeromoniasis": {"Red Disease": 0, "Aeromoniasis": 94, "Gill Disease": 3, "Saprolegniasis": 0, "Healthy Fish": 3, "Parasitic Disease": 0, "White Tail Disease": 0},
    "Gill Disease": {"Red Disease": 2, "Aeromoniasis": 0, "Gill Disease": 95, "Saprolegniasis": 0, "Healthy Fish": 0, "Parasitic Disease": 1, "White Tail Disease": 2},
    "Saprolegniasis": {"Red Disease": 5, "Aeromoniasis": 1, "Gill Disease": 0, "Saprolegniasis": 90, "Healthy Fish": 1, "Parasitic Disease": 3, "White Tail Disease": 0},
    "Healthy Fish": {"Red Disease": 1, "Aeromoniasis": 0, "Gill Disease": 0, "Saprolegniasis": 0, "Healthy Fish": 99, "Parasitic Disease": 0, "White Tail Disease": 0},
    "Parasitic Disease": {"Red Disease": 0, "Aeromoniasis": 3, "Gill Disease": 7, "Saprolegniasis": 1, "Healthy Fish": 4, "Parasitic Disease": 83, "White Tail Disease": 2},
    "White Tail Disease": {"Red Disease": 4, "Aeromoniasis": 0, "Gill Disease": 3, "Saprolegniasis": 2, "Healthy Fish": 5, "Parasitic Disease": 1, "White Tail Disease": 85}
}


df = pd.DataFrame(confusion_matrix).T  # Transpose for correct orientation
plt.figure(figsize=(25, 25))
sns.heatmap(df, annot=True)
plt.title("Confusion Matrix for MobileNetV2 based Classifier Model")
plt.ylabel("Predicted Disease")
plt.xlabel("Actual Disease")
plt.show()


cm = ConfusionMatrix(matrix = confusion_matrix)
print(cm)
