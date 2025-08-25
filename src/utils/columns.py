"""
Feature schema for NSL-KDD (KDDTrain+/KDDTest+).

There are 41 input features:
- 3 categorical: protocol_type, service, flag
- 38 numeric: the rest
"""

CATEGORICAL = ["protocol_type", "service", "flag"]

NUMERIC = [
    "duration",
    "src_bytes",
    "dst_bytes",
    "land",
    "wrong_fragment",
    "urgent",
    "hot",
    "num_failed_logins",
    "logged_in",
    "num_compromised",
    "root_shell",
    "su_attempted",
    "num_root",
    "num_file_creations",
    "num_shells",
    "num_access_files",
    "num_outbound_cmds",
    "is_host_login",
    "is_guest_login",
    "count",
    "srv_count",
    "serror_rate",
    "srv_serror_rate",
    "rerror_rate",
    "srv_rerror_rate",
    "same_srv_rate",
    "diff_srv_rate",
    "srv_diff_host_rate",
    "dst_host_count",
    "dst_host_srv_count",
    "dst_host_same_srv_rate",
    "dst_host_diff_srv_rate",
    "dst_host_same_src_port_rate",
    "dst_host_srv_diff_host_rate",
    "dst_host_serror_rate",
    "dst_host_srv_serror_rate",
    "dst_host_rerror_rate",
    "dst_host_srv_rerror_rate",
]

ALL_FEATURES = ["duration", "protocol_type", "service", "flag",
                "src_bytes", "dst_bytes", "land", "wrong_fragment",
                "urgent", "hot", "num_failed_logins", "logged_in",
                "num_compromised", "root_shell", "su_attempted", "num_root",
                "num_file_creations", "num_shells", "num_access_files",
                "num_outbound_cmds", "is_host_login", "is_guest_login",
                "count", "srv_count", "serror_rate", "srv_serror_rate",
                "rerror_rate", "srv_rerror_rate", "same_srv_rate",
                "diff_srv_rate", "srv_diff_host_rate", "dst_host_count",
                "dst_host_srv_count", "dst_host_same_srv_rate",
                "dst_host_diff_srv_rate", "dst_host_same_src_port_rate",
                "dst_host_srv_diff_host_rate", "dst_host_serror_rate",
                "dst_host_srv_serror_rate", "dst_host_rerror_rate",
                "dst_host_srv_rerror_rate"]

LABEL = "label"
DIFFICULTY = "difficulty"

CSV_COLUMNS = ALL_FEATURES + [LABEL, DIFFICULTY]


# Map original NSL-KDD labels to coarse attack categories
# Source categories commonly used for NSL-KDD: dos, probe, r2l, u2r, normal
ATTACK_TYPE_MAP = {
    # DoS
    "apache2": "dos", "back": "dos", "neptune": "dos", "mailbomb": "dos",
    "processtable": "dos", "smurf": "dos", "teardrop": "dos", "udpstorm": "dos",
    "worm": "dos", "land": "dos", "pod": "dos",
    # Probe
    "ipsweep": "probe", "mscan": "probe", "nmap": "probe",
    "portsweep": "probe", "saint": "probe", "satan": "probe",
    # R2L
    "ftp_write": "r2l", "guess_passwd": "r2l", "httptunnel": "r2l",
    "imap": "r2l", "multihop": "r2l", "named": "r2l", "phf": "r2l",
    "sendmail": "r2l", "snmpgetattack": "r2l", "snmpguess": "r2l",
    "spy": "r2l", "warezclient": "r2l", "warezmaster": "r2l", "xlock": "r2l",
    "xsnoop": "r2l",
    # U2R
    "buffer_overflow": "u2r", "loadmodule": "u2r", "perl": "u2r",
    "ps": "u2r", "rootkit": "u2r", "sqlattack": "u2r", "xterm": "u2r",
    # Normal
    "normal": "normal",
}
COARSE_CLASSES = ["normal", "dos", "probe", "r2l", "u2r"]
