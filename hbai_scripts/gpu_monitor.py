import subprocess
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
import time

def check_gpu_usage(threshold=5):
    # Run the nvidia-smi command and get the output
    result = subprocess.run(['nvidia-smi', '--query-gpu=utilization.gpu', '--format=csv,noheader,nounits'], 
                            stdout=subprocess.PIPE)
    output = result.stdout.decode('utf-8').strip()
    gpu_usages = [int(x) for x in output.split('\n')]

    # Check if all GPUs are below the threshold
    return all(usage < threshold for usage in gpu_usages)

def send_email(subject, body, to_email, from_email, smtp_server, smtp_port, smtp_user, smtp_password):
    msg = MIMEMultipart()
    msg['From'] = from_email
    msg['To'] = to_email
    msg['Subject'] = subject

    msg.attach(MIMEText(body, 'plain'))

    server = smtplib.SMTP(smtp_server, smtp_port)
    server.starttls()
    server.login(smtp_user, smtp_password)
    text = msg.as_string()
    server.sendmail(from_email, to_email, text)
    server.quit()

if __name__ == "__main__":
    # Email configuration
    TO_EMAIL = 'your_email@gmail.com'
    FROM_EMAIL = 'your_email@gmail.com'
    SMTP_SERVER = 'smtp.gmail.com'
    SMTP_PORT = 587
    SMTP_USER = 'your_email@gmail.com'
    SMTP_PASSWORD = 'your_app_password'  # Use an app-specific password

    # Loop to continuously monitor GPU usage
    while True:
        if check_gpu_usage():
            subject = "GPU Usage Alert"
            body = "The GPUs on the server are currently not being used."
            send_email(subject, body, TO_EMAIL, FROM_EMAIL, SMTP_SERVER, SMTP_PORT, SMTP_USER, SMTP_PASSWORD)

        # Wait for 10 minutes before checking again (adjust as needed)
        time.sleep(600)  # 600 seconds = 10 minutes
