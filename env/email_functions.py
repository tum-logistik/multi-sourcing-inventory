import smtplib
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from common.variables import *
import platform
import socket
from email.mime.application import MIMEApplication
from email.mime.multipart import MIMEMultipart
from os.path import basename


def send_email(sender_address = SENDER_EMAIL_ADDRESS,
        sender_pass = SENDER_EMAIL_PASSWORD,
        recipient_email_address = RECIPIENT_EMAIL_ADDRESS,
        file_id = "dummy_id",
        mail_content = "dummy message",
        files = [],
        subject_header = 'DS-Sim: '
    ):

    host_name = socket.gethostname()

    if not sender_address or not sender_pass or not recipient_email_address:
        print("No email fields specified in yaml config file! No email sent.")
        return False
    
    message = MIMEMultipart()
    message['From'] = sender_address
    message['To'] = recipient_email_address
    message['Subject'] = subject_header + host_name + " / " + file_id # The subject line
    
    ## The body and the attachments for the mail
    message.attach(MIMEText(mail_content, 'plain'))

    for f in files:
        with open(f, "rb") as fil:
            part = MIMEApplication(
                fil.read(),
                Name=basename(f)
            )
        # After the file is closed
        part['Content-Disposition'] = 'attachment; filename="%s"' % basename(f)
        message.attach(part)

    # Create SMTP session for sending the mail
    session = smtplib.SMTP('smtp.gmail.com', 587) # use gmail with port
    session.starttls() #enable security
    session.login(sender_address, sender_pass) #login with mail_id and password
    text = message.as_string()
    session.sendmail(sender_address, recipient_email_address, text)
    session.quit()

    print("message sent to: " +  message['To'])
    print('Subject: ' + message['Subject'])

    
    


    return True

# The mail addresses and password



#Setup the MIME




