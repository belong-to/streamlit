import mysql.connector

# 配置数据库连接信息
config = {
    'user': 'root',
    'password': '123456',
    'host': '127.0.0.1',
    'database': 'defect',
    'raise_on_warnings': True
}

# 创建连接
connection = mysql.connector.connect(**config)

try:
    # connection = mysql.connector.connect(**config)
    if connection.is_connected():
        print("Connection successful")
    else:
        print("Connection unsuccessful")
except mysql.connector.Error as err:
    print(f"Error: {err}")
finally:
    if connection.is_connected():
        connection.close()
        print("Connection closed")