import pymysql


class SQL:
    def __init__(self, user="root", pw="root", port=3306, sql_path=""):
        # Connect to mySQL Server and create a cursor

        self.conn = pymysql.connect(host="localhost", port=port, user=user, passwd=pw, db="mysql", charset="utf8")
        self.cur = self.conn.cursor()

        self.use_db()
        self.run_file(sql_path)

    def get_table(self, table):
        if table not in {"analog_values", "measurements"}:
            return
        self.cur.execute("SELECT COUNT(DISTINCT name) FROM " + table)
        n, = self.cur.fetchone()
        self.cur.execute("SELECT COUNT(DISTINCT time) FROM " + table)
        t, = self.cur.fetchone()
        self.cur.execute("SELECT value FROM " + table + " ORDER BY time, name")
        self.data = []
        for i in range(t):
            self.data.append([])
            for j in range(n):
                val, = self.cur.fetchone()
                self.data[i].append(val)

        self.cur.execute("SELECT DISTINCT name FROM analog_values ORDER by name")
        headers = self.cur.fetchall()
        headers = [a[0] for a in headers]
        return self.data, headers

    def use_db(self):
        # Delete, Create and Use case9bus database
        self.cur.execute(
            "DROP DATABASE IF EXISTS case9bus;"
            "CREATE DATABASE case9bus;"
            "USE case9bus;"
        )

    def run_file(self, file):
        # Execute all commands in sql file

        with open(file) as f:
            cmd = ""
            for line in f.readlines():
                # Skip comments and empty lines
                if line.strip()[:2] == "--" or line.strip() == "":
                    continue

                cmd += line.strip()
                if cmd[-1] == ";":
                    self.cur.execute(cmd)
                    cmd = ""


    def close_sql(self):
        # Disconnect from mysql server
        self.cur.close()
        self.conn.close()


if __name__ == "__main__":
    sql = SQL(sql_path=r"..\..\assignment2code_test.sql")
    sql.get_table("measurements")
    sql.close_sql()