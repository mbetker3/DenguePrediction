
import seaborn as sns
import matplotlib.pyplot as plt


def plotGraphs(data):
    #plotting graphs
    sns.countplot(x="Gender", data=data, palette="Set2")
    plt.title("Gender Distribution of Dengue Patients")
    plt.show()

    sns.countplot(x="Gender", hue="Outcome", data=data, palette="husl")
    plt.title("Outcome by Gender")
    plt.show()

    sns.boxplot(x="Outcome", y="Age", data=data, palette="Set3")
    plt.title("Age Distribution by Dengue Outcome")
    plt.show()

    sns.countplot(x="AreaType", hue="Outcome", data=data, palette="viridis")
    plt.title("Developed vs Undeveloped Areas by Outcome")
    plt.show()

    sns.heatmap(data[["NS1", "IgG", "IgM", "Outcome"]].corr(), annot=True, cmap="coolwarm")
    plt.title("Correlation Between Antibody Tests and Outcome")
    plt.show()