from langchain_community.utilities import SQLDatabase
from langchain.chains import create_sql_query_chain
from langchain_openai import ChatOpenAI


def main():
    # ====== 1. Thông tin kết nối MySQL ======
    USER = "root"          # Tên user MySQL
    PASSWORD = "12345abc"  # Mật khẩu MySQL
    HOST = "localhost"     # Địa chỉ server
    PORT = 3306            # Cổng mặc định MySQL
    DB_NAME = "flight_booking_db"

    # ====== 2. Kết nối database ======
    uri = f"mysql+pymysql://{USER}:{PASSWORD}@{HOST}:{PORT}/{DB_NAME}"
    db = SQLDatabase.from_uri(uri)

    # ====== 3. Tạo LLM ======
    llm = ChatOpenAI(
        model="gpt-4o-mini",
        temperature=0,
        api_key="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"
    )

    # ====== 4. Tạo chain sinh SQL ======
    write_query = create_sql_query_chain(llm, db)

    print("🚀 Kết nối thành công! Bạn có thể đặt câu hỏi (tiếng Anh hoặc tiếng Việt).")
    while True:
        question = input("\n❓ Câu hỏi (gõ 'exit' để thoát): ")
        if question.lower() in ["exit", "quit"]:
            break
        try:
            # Sinh SQL từ câu hỏi
            sql_query = write_query.invoke({"question": question})
            print("📝 SQL sinh ra (raw):", sql_query)

            # ====== Làm sạch SQL ======
            if sql_query.startswith("SQLQuery:"):
                sql_query = sql_query.replace("SQLQuery:", "").strip()
            # bỏ wrapper ```sql ... ```
            sql_query = sql_query.replace("```sql", "").replace("```", "").strip()

            print("✅ SQL sau khi làm sạch:", sql_query)

            # Thực thi SQL
            result = db.run(sql_query)
            print("💡 Kết quả:", result)
        except Exception as e:
            print("⚠️ Lỗi:", e)


if __name__ == "__main__":
    main()