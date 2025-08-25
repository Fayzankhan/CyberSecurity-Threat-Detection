from setuptools import setup, find_packages

setup(
    name="cyber-threat-detector",
    version="1.0.0",
    packages=find_packages(),
    install_requires=[
        "pandas>=2.0",
        "numpy>=1.24",
        "scikit-learn>=1.3",
        "joblib>=1.3",
        "fastapi>=0.110",
        "uvicorn[standard]>=0.27",
        "streamlit>=1.34",
        "plotly>=1.34",
        "requests>=2.31",
    ],
    python_requires=">=3.10",
)