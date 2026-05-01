def create_risk(row):
    score = 0

    if row['BMI_Value'] > 30:
        score += 2
    elif row['BMI_Value'] > 25:
        score += 1

    if row['Stress Level'] > 7:
        score += 2
    elif row['Stress Level'] > 4:
        score += 1

    if row['Sleep Duration'] < 5:
        score += 2
    elif row['Sleep Duration'] < 7:
        score += 1

    if row['Physical Activity Level'] < 20:
        score += 2
    elif row['Physical Activity Level'] < 50:
        score += 1

    if score >= 5:
        return "High"
    elif score >= 2:
        return "Medium"
    else:
        return "Low"