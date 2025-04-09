def bigfive_to_text(bigfive):
    extraversion = bigfive.get("extraversion", 0.5)
    neuroticism = bigfive.get("neuroticism", 0.5)
    openness = bigfive.get("openness", 0.5)
    agreeableness = bigfive.get("agreeableness", 0.5)
    conscientiousness = bigfive.get("conscientiousness", 0.5)
    
    description = []
    
    # Extraversion
    if extraversion > 0.7:
        description.append("highly extraverted")
    elif extraversion < 0.3:
        description.append("introverted")
    else:
        description.append("moderately extraverted")
    
    # Neuroticism (inverse for stability)
    if neuroticism > 0.7:
        description.append("prone to anxiety")
    elif neuroticism < 0.3:
        description.append("emotionally stable")
    else:
        description.append("moderately emotionally stable")
        
    # Openness
    if openness > 0.7:
        description.append("very open to new experiences")
    elif openness < 0.3:
        description.append("more conventional")
    else:
        description.append("moderately open")
    
    # Agreeableness
    if agreeableness > 0.7:
        description.append("highly cooperative and empathetic")
    elif agreeableness < 0.3:
        description.append("more competitive")
    else:
        description.append("fairly agreeable")
    
    # Conscientiousness
    if conscientiousness > 0.7:
        description.append("very conscientious")
    elif conscientiousness < 0.3:
        description.append("laid-back")
    else:
        description.append("moderately conscientious")
    
    return ", ".join(description)
