import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

# 1. Charger les données
data = pd.read_csv("train.csv")

# 2. Créer la feature FamilySize
data["FamilySize"] = data["SibSp"] + data["Parch"] + 1

# 3. Extraire le titre depuis le nom
def get_title(name):
    return name.split(",")[1].split(".")[0].strip()

data["Title"] = data["Name"].apply(get_title)

# 4. Regrouper les titres rares
rare_titles = ['Lady','Monsieur','Madame']
data["Title"] = data["Title"].replace(rare_titles, "Rare")

# 5. Remplir les valeurs manquantes
data["Age"].fillna(data["Age"].median(), inplace=True)
data["Embarked"].fillna(data["Embarked"].mode()[0], inplace=True)

# 6. Supprimer la colonne Name (inutile)
data = data.drop(columns=["Name", "Ticket", "Cabin", "PassengerId"])

# 7. Transformer les variables catégorielles en numériques (one-hot)
data = pd.get_dummies(data, columns=["Sex", "Embarked", "Title"], drop_first=True)

# 8. Séparer les données en features X et cible y
X = data.drop("Survived", axis=1)
y = data["Survived"]

# 9. Diviser en jeu d'entraînement et de test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 10. Créer et entraîner le modèle RandomForest
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# 11. Faire des prédictions sur le test
y_pred = model.predict(X_test)

# 12. Afficher les résultats
print("Accuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))
