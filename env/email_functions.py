import smtplib
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from common.variables import *

def send_email(sender_address = SENDER_EMAIL_ADDRESS,
    sender_pass = SENDER_EMAIL_PASSWORD,
    recipient_email_address = RECIPIENT_EMAIL_ADDRESS):

    mail_content = '''Experimental Results are here for Larkin:
    '''

    if not sender_address or not sender_pass or not recipient_email_address:
        print("No email fields specified in yaml config file! No email sent.")
        return False
    
    message = MIMEMultipart()
    message['From'] = sender_address
    message['To'] = recipient_email_address

    message['Subject'] = 'A test mail sent by Python. It has an attachment.'   #The subject line
    #The body and the attachments for the mail
    message.attach(MIMEText(mail_content, 'plain'))

    #Create SMTP session for sending the mail
    session = smtplib.SMTP('smtp.gmail.com', 587) #use gmail with port
    session.starttls() #enable security
    session.login(sender_address, sender_pass) #login with mail_id and password
    text = message.as_string()
    session.sendmail(sender_address, recipient_email_address, text)
    session.quit()
    print('Mail Sent')

    return 1

# The mail addresses and password



#Setup the MIME




