import streamlit as st
import sqlite3
import duckdb
import pandas as pd
import time
import numpy as np

# -------------------------
# Utility functions
# -------------------------

def get_small_data():
    return pd.DataFrame({
        "id": range(1, 11),
        "name": ["Alice", "Bob", "Charlie", "David", "Eva", 
                 "Frank", "Grace", "Hannah", "Ivan", "Julia"],
        "age": [25, 32, 37, 29, 41, 35, 28, 39, 45, 31],
        "department": ["HR", "IT", "Finance", "IT", "HR",
                       "Finance", "IT", "HR", "Finance", "IT"]
    })

def init_sqlite(df):
    conn = sqlite3.connect(":memory:")
    df.to_sql("employees", conn, index=False, if_exists="replace")
    return conn

def init_duckdb(df):
    conn = duckdb.connect(":memory:")
    conn.register("employees", df)
    return conn

def run_query(conn, query, engine="sqlite"):
    start = time.time()
    if engine == "sqlite":
        result = pd.read_sql_query(query, conn)
    else:
        result = conn.execute(query).df()
    duration = (time.time() - start) * 1000
    return result, duration

# -------------------------
# Streamlit Tabs
# -------------------------

tab1, tab2, tab3 = st.tabs(["📖 Overview", "🧪 Small Example", "📊 Big Data Demo"])

# -------------------------
# Tab 1 — Overview
# -------------------------
with tab1:
    st.title("SQLite vs DuckDB vs Blob Storage — Overview")

    # ---- SQLite Section ----
    with st.container():
        st.markdown("### 🗄️ SQLite", unsafe_allow_html=True)
        st.markdown("""
        **About**  
        SQLite is a lightweight, file-based relational database engine.  
        It requires no separate server process and stores the entire database 
        in a single file. It’s widely used in mobile apps, embedded devices, 
        and desktop applications.
        """)

        with st.expander("📌 Use Cases"):
            st.markdown("""
            - Mobile apps & desktop software  
            - Small/medium local databases  
            - Config/metadata storage  
            - Apps needing ACID compliance  
            """)

        with st.expander("✅ Pros"):
            st.markdown("""
            - 💡 Very lightweight & widely supported  
            - ⚡ Zero setup (just a file or `:memory:`)  
            - 🔒 Strong transactional guarantees (ACID)  
            - 📊 Great for small row-based workloads  
            """)

        with st.expander("⚠️ Cons"):
            st.markdown("""
            - ❌ Not optimized for analytics  
            - 🐢 Slower for big aggregations/joins  
            - 🚫 No parallel execution  
            """)

    st.markdown("---")  # divider between cards

    # ---- DuckDB Section ----
    with st.container():
        st.markdown("### 🦆 DuckDB", unsafe_allow_html=True)
        st.markdown("""
        **About**  
        DuckDB is an in-process SQL **OLAP** database optimized for analytics.  
        It integrates seamlessly with Python, R, and data science workflows, 
        and is often called *“SQLite for analytics”*.  
        """)

        # Tooltip-like OLAP explanation
        with st.expander("ℹ️ What does OLAP mean?"):
            st.markdown("""
            **OLAP (Online Analytical Processing)** databases are optimized for:  
            - Large-scale data analysis  
            - Aggregations, grouping, and summarization  
            - Fast columnar reads on big datasets  

            In contrast, **OLTP (Online Transaction Processing)** databases 
            like SQLite are optimized for:  
            - Small transactions  
            - Inserts/updates/deletes  
            - Row-based workloads  
            """)

        with st.expander("📌 Use Cases"):
            st.markdown("""
            - Data analysis & BI workloads  
            - Processing large CSV/Parquet files  
            - Quick prototyping for data science  
            - In-memory analytics like "SQLite for OLAP"  
            """)

        with st.expander("✅ Pros"):
            st.markdown("""
            - ⚡ Columnar storage (fast aggregations)  
            - 🚀 Blazing fast on millions/billions of rows  
            - 📂 Reads/writes Parquet natively  
            - 🐍 Great integration with Python & Pandas  
            - ☁️ Can query S3 / Azure Blob directly  
            """)

        with st.expander("⚠️ Cons"):
            st.markdown("""
            - ❌ Heavier than SQLite for tiny data  
            - 🆕 Relatively new (not as battle-tested)  
            - 🔧 Fewer production-ready transactional features  
            """)

    st.markdown("---")  # divider between cards

    # ---- Blob Storage Section ----
    with st.container():
        st.markdown("### ☁️ Blob Storage (S3 & Azure Blob)", unsafe_allow_html=True)
        st.markdown("""
        **About**  
        Blob storage (**Binary Large Object storage**) is designed for storing unstructured data:  
        files, images, backups, videos, and analytical datasets.  
        Instead of rows and columns, it stores objects with metadata, making it the backbone of cloud data lakes.  
        """)

        with st.expander("📌 Examples"):
            st.markdown("""
            - **Amazon S3 (Simple Storage Service)**  
              Industry standard object storage in AWS, organizes data in *buckets*.  

            - **Azure Blob Storage**  
              Microsoft’s equivalent, with hot/cool/archive tiers depending on access needs.  
            """)

        with st.expander("✅ Pros"):
            st.markdown("""
            - ☁️ Infinitely scalable for large datasets  
            - 💰 Cost-effective (pay as you go)  
            - 🔒 Integrated with cloud IAM/security  
            - 🌍 Globally available and replicated  
            """)

        with st.expander("⚠️ Cons"):
            st.markdown("""
            - ❌ Higher latency than local databases  
            - ⚠️ Requires APIs/SDKs for access (not a simple file open)  
            - 🔗 Eventual consistency (not always ACID)  
            - 💸 Frequent small reads/writes can be expensive  
            """)

        with st.expander("🔄 Interaction with SQLite & DuckDB"):
            st.markdown("""
            **SQLite**  
            - Does not integrate directly with S3/Azure Blob.  
            - Expects a local `.db` file (must download first).  

            **DuckDB**  
            - Natively integrates with S3/Azure Blob.  
            - Can query Parquet/CSV files in blob storage directly with SQL.  
            - Makes it ideal for cloud-native analytics pipelines.  
            """)




# -------------------------
# Tab 2 — Small Example
# -------------------------
with tab2:
    st.title("Small Data Demo")

    df = get_small_data()
    st.subheader("Sample Data")
    st.dataframe(df.head())

    query = st.text_area(
        "Enter a SQL query (works on both):",
        "SELECT department, AVG(age) as avg_age FROM employees GROUP BY department;"
    )

    if st.button("Run Query on Small Data"):
        # ----- SQLite total time -----
        start = time.time()
        sqlite_conn = init_sqlite(df)
        sqlite_result, _ = run_query(sqlite_conn, query, "sqlite")
        sqlite_total = (time.time() - start) * 1000

        # ----- DuckDB total time -----
        start = time.time()
        duckdb_conn = init_duckdb(df)
        duckdb_result, _ = run_query(duckdb_conn, query, "duckdb")
        duckdb_total = (time.time() - start) * 1000

        col1, col2 = st.columns(2)

        with col1:
            st.write("### SQLite Results")
            st.dataframe(sqlite_result)
            st.caption(f"⏱ Total time: {sqlite_total:.2f} ms")

        with col2:
            st.write("### DuckDB Results")
            st.dataframe(duckdb_result)
            st.caption(f"⏱ Total time: {duckdb_total:.2f} ms")

        # ----- Runtime comparison chart -----
        st.subheader("Total Runtime Comparison (ms)")
        runtime_df = pd.DataFrame({
            "Engine": ["SQLite", "DuckDB"],
            "Total Time (ms)": [sqlite_total, duckdb_total]
        }).set_index("Engine")

        st.bar_chart(runtime_df, use_container_width=True)


# -------------------------
# Tab 3 — Big Data Demo
# -------------------------
with tab3:
    st.title("Big Data Benchmark")

    rows = st.slider("Number of rows to generate", 1_000_000, 10_000_000, 1_000_000, step=1_000_000)

    query = st.text_area(
        "Enter a SQL query for big data:",
        "SELECT department, AVG(salary) as avg_salary, COUNT(*) as cnt FROM employees GROUP BY department;"
    )

    if st.button("Run Query on Big Data"):
        # ----- SQLite total time -----
        start = time.time()
        np.random.seed(42)
        big_df = pd.DataFrame({
            "id": np.arange(rows),
            "age": np.random.randint(18, 65, size=rows),
            "salary": np.random.randint(40_000, 120_000, size=rows),
            "department": np.random.choice(["HR", "IT", "Finance", "Marketing"], size=rows)
        })
        sqlite_conn = init_sqlite(big_df)
        sqlite_result, _ = run_query(sqlite_conn, query, "sqlite")
        sqlite_total = (time.time() - start) * 1000

        # ----- DuckDB total time -----
        start = time.time()
        np.random.seed(42)
        big_df = pd.DataFrame({
            "id": np.arange(rows),
            "age": np.random.randint(18, 65, size=rows),
            "salary": np.random.randint(40_000, 120_000, size=rows),
            "department": np.random.choice(["HR", "IT", "Finance", "Marketing"], size=rows)
        })
        duckdb_conn = init_duckdb(big_df)
        duckdb_result, _ = run_query(duckdb_conn, query, "duckdb")
        duckdb_total = (time.time() - start) * 1000

        # ----- Show results -----
        col1, col2 = st.columns(2)

        with col1:
            st.write("### SQLite Results")
            st.dataframe(sqlite_result)
            st.caption(f"⏱ Total time: {sqlite_total:.2f} ms")

        with col2:
            st.write("### DuckDB Results")
            st.dataframe(duckdb_result)
            st.caption(f"⏱ Total time: {duckdb_total:.2f} ms")

        # ----- Runtime comparison chart -----
        st.subheader("Total Runtime Comparison (ms)")
        runtime_df = pd.DataFrame({
            "Engine": ["SQLite", "DuckDB"],
            "Total Time (ms)": [sqlite_total, duckdb_total]
        }).set_index("Engine")

        st.bar_chart(runtime_df, use_container_width=True)
