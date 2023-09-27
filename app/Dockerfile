FROM python:3.8-slim
RUN apt-get update && apt-get install -y \
    build-essential \
    libmariadb-dev-compat \
    libmariadb-dev \
 && apt-get clean
RUN pip install --upgrade pip
# Set the working directory
WORKDIR /app

# Copy the requirements file into the container
COPY ./requirements.txt .

# Manually specify MySQL client library flags
ENV MYSQLCLIENT_CFLAGS="-I/usr/include/mariadb"
ENV MYSQLCLIENT_LDFLAGS="-L/usr/lib/x86_64-linux-gnu -lmysqlclient"

# Install Flask and other dependencies
RUN pip install mysql-connector-python
RUN pip install -r requirements.txt

# Copy the application code into the container
COPY . .

# Expose the port that Flask will run on
EXPOSE 3000

# Define the command to run your application
CMD ["python", "app.py"]
