About Dataset
EMAIL SPAM CLASSIFICATION

Introduction
Email spam, also known as junk email, is unsolicited messages sent in bulk by email. Spam can be an annoyance, consume network bandwidth, and even contain malicious links or phishing attempts. Accurate classification of email as spam (junk) or ham (legitimate) is crucial for email service providers and users to maintain a clean inbox and ensure security. This dataset provides a collection of emails with their corresponding labels to aid in building machine learning models for spam detection.

Data Preparation

Source
This dataset is a sample collection of emails, intended for building and evaluating text classification models, particularly for spam detection.

Variables
The dataset contains the following attributes for each email record:

• subject: The subject line of the email. (Text)
• body: The main content/body of the email. (Text)
• from: The sender's email address. (Text/Nominal)
• label: The classification label of the email. 'ham' indicates a legitimate email, and 'spam' indicates a junk/unsolicited email. (Target Variable - Nominal)
