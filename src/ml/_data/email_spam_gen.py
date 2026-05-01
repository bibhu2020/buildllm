import pandas as pd
import numpy as np
import random
import uuid
from datetime import datetime, timedelta

def generate_very_dirty_spam_data(n=3600):
    spam_phrases = [
        "WINNER! Claim your prize now at <a href='http://malware.com'>here</a>",
        "Free money!!! Just click <b>&nbsp;this link&nbsp;</b>",
        "Your account has been suspended. Please login to verify.",
        "Cheap meds, no prescription needed!!!",
        "Meeting at 5pm? Let me know.",
        "How have you been? It has been a long time.",
        "Attached is the invoice for last month.",
        "Hello, we are interested in your profile.",
        "¡Urgente! Necesitamos su respuesta inmediata.",
        "Bonjour, voici les documents demandés."
    ]
    
    data = []
    for i in range(n):
        email_id = str(uuid.uuid4())
        sender = f"user{random.randint(1, 1000)}@example.com"
        subject = f"Subject {i}"
        
        # Content with HTML noise
        content = random.choice(spam_phrases)
        if random.random() < 0.3:
            content = f"<div>{content}</div><br><p>Best regards,</p>"
            
        # Date chaos
        base_date = datetime.now() - timedelta(days=random.randint(0, 365))
        fmt = random.choice(['iso', 'unix', 'mdy', 'dmy'])
        if fmt == 'iso': date = base_date.isoformat()
        elif fmt == 'unix': date = str(int(base_date.timestamp()))
        elif fmt == 'mdy': date = base_date.strftime('%m/%d/%Y')
        else: date = base_date.strftime('%d-%m-%Y')
        
        # Target Label chaos
        label_raw = random.choice(['spam', 'ham'])
        if label_raw == 'spam':
            target_label = random.choice(['spam', 'SPAM', '1', 's_p_a_m'])
        else:
            target_label = random.choice(['ham', 'Ham', '0', 'Legit'])
            
        priority = random.choice(['High', 'Medium', 'Low', 'null', 'N/A', 'Urgent!'])
        attachments = random.choice([0, 1, 2, 'None', 'null', -1])
        links = random.choice([0, 1, 5, 10, 100]) # Some outliers
        
        data.append([email_id, sender, subject, content, date, priority, attachments, links, target_label, "v1.0"])
        
    cols = ["Email_ID", "Sender_Email", "Subject_Line", "Message-Content", "Date", "Priority", "Attachments", "Links_Count", "Target_Label", "Version"]
    df = pd.DataFrame(data, columns=cols)
    
    # Add duplicates
    df_dupes = df.sample(100)
    df = pd.concat([df, df_dupes], ignore_index=True)
    
    df.to_csv('email_spam.csv', index=False)
    print(f"Generated {len(df)} records in email_spam.csv")

if __name__ == "__main__":
    generate_very_dirty_spam_data(3600)
