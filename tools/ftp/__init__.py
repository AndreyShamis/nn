from ftplib import FTP
ftp = FTP('ftp.freehosting.anshamis.com')     # connect to host, default port
ftp.login(user="nn@ftp.freehosting.anshamis.com", passwd="polkalol")  # user anonymous, passwd anonymous@

ftp.cwd('nn')               # change into "debian" directory
ftp.retrlines('LIST')           # list directory contents


ftp.quit()