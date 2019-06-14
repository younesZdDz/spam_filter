import os
import tarfile
from six.moves import urllib
import email
import email.policy
import os


def fetch_spam_data(spam_url, ham_url, spam_path ):
    if not os.path.isdir(spam_path):
        os.makedirs(spam_path)
    for filename, url in (("ham.tar.bz2", ham_url), ("spam.tar.bz2", spam_url)):
        path = os.path.join(spam_path, filename)
        if not os.path.isfile(path):
            urllib.request.urlretrieve(url, path)
        tar_bz2_file = tarfile.open(path)
        tar_bz2_file.extractall(path=spam_path)
        tar_bz2_file.close()



SPAM_PATH = os.path.join("model", "datasets", "spam")

def load_email(is_spam, filename, spam_path=SPAM_PATH):
    directory = "spam" if is_spam else "easy_ham"
    with open(os.path.join(spam_path, directory, filename), "rb") as f:
        return email.parser.BytesParser(policy=email.policy.default).parse(f)


def email_to_text(email):
    for part in email.walk():
        ctype= part.get_content_type()
        if not ctype in ('text/plain', 'text/html'):
            continue 
        try:
            content= part.get_content()
        except:
            content = str(part.get_payload())
        return content


def get_email_structure(email):
    if isinstance(email, str):
        return 'text/plain'
    payload = email.get_payload()
    if isinstance(payload, list):
        return ", ".join([
            get_email_structure(sub_email)
            for sub_email in payload
        ])
    else:
        return email.get_content_type()