import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st

# ── PAGE CONFIG ──────────────────────────────────────────────
st.set_page_config(
    page_title="Olist E-Commerce Dashboard",
    page_icon="🛒",
    layout="wide"
)

# ── LOAD DATA ─────────────────────────────────────────────────
@st.cache_data
def load_data():
    base = "data/"
    df_order_items    = pd.read_csv(base + "order_items_dataset.csv")
    df_product        = pd.read_csv(base + "products_dataset.csv")
    df_order_payments = pd.read_csv(base + "order_payments_dataset.csv")
    df_order          = pd.read_csv(base + "orders_dataset.csv")
    df_customer       = pd.read_csv(base + "customers_dataset.csv")
    df_kategori       = pd.read_csv(base + "product_category_name_translation.csv")

    df_kategori.columns = df_kategori.columns.str.replace('\ufeff', '', regex=False)

    datetime_cols = [
        'order_purchase_timestamp', 'order_approved_at',
        'order_delivered_carrier_date', 'order_delivered_customer_date',
        'order_estimated_delivery_date'
    ]
    for col in datetime_cols:
        df_order[col] = pd.to_datetime(df_order[col])

    df_product = df_product.dropna(subset=['product_category_name'])
    cols_dimensi = ['product_weight_g', 'product_length_cm', 'product_height_cm', 'product_width_cm']
    for col in cols_dimensi:
        df_product[col] = df_product[col].fillna(df_product[col].median())

    orders_delivered = df_order[df_order['order_status'] == 'delivered'].copy()

    return df_order_items, df_product, df_order_payments, df_order, df_customer, df_kategori, orders_delivered


@st.cache_data
def process_q1_base(df_order_items, df_product, df_kategori, orders_delivered):
    orders_q1 = orders_delivered[
        (orders_delivered['order_purchase_timestamp'] >= '2017-01-01') &
        (orders_delivered['order_purchase_timestamp'] < '2018-09-01')
    ].copy()

    df = df_order_items.merge(
        df_product[['product_id', 'product_category_name']],
        on='product_id', how='left'
    ).merge(
        df_kategori, on='product_category_name', how='left'
    ).merge(
        orders_q1[['order_id']], on='order_id', how='inner'
    )
    df['product_category_name_english'] = df['product_category_name_english'].fillna('unknown')

    pivot = df.groupby('product_category_name_english').agg(
        total_items_sold=('order_item_id', 'count'),
        total_revenue=('price', 'sum'),
        avg_price=('price', 'mean'),
    ).round(2).sort_values('total_items_sold', ascending=False)

    pivot['pct_share'] = (pivot['total_items_sold'] / pivot['total_items_sold'].sum() * 100).round(2)
    return pivot


@st.cache_data
def process_q2_base(orders_delivered, df_order_payments):
    payment_agg = df_order_payments.groupby('order_id').agg(
        total_payment=('payment_value', 'sum')
    ).reset_index()

    df_revenue = orders_delivered.merge(payment_agg, on='order_id', how='left')
    df_revenue = df_revenue[
        (df_revenue['order_purchase_timestamp'] >= '2017-01-01') &
        (df_revenue['order_purchase_timestamp'] < '2018-01-01')
    ].copy()
    df_revenue['year_month'] = df_revenue['order_purchase_timestamp'].dt.to_period('M')

    pivot = df_revenue.groupby('year_month').agg(
        total_revenue=('total_payment', 'sum'),
        order_count=('order_id', 'count'),
        avg_order_value=('total_payment', 'mean')
    ).round(2)
    return pivot, payment_agg, df_revenue


@st.cache_data
def process_q3(orders_delivered, df_customer, payment_agg):
    customers = orders_delivered.merge(
        df_customer[['customer_id', 'customer_unique_id']],
        on='customer_id', how='left'
    ).merge(payment_agg, on='order_id', how='left')

    frekuensi = customers.groupby('customer_unique_id').agg(
        jumlah_order=('order_id', 'count'),
        avg_spending=('total_payment', 'mean')
    ).reset_index()

    frekuensi['segment'] = frekuensi['jumlah_order'].apply(
        lambda x: 'returning' if x > 1 else 'new'
    )

    pivot = frekuensi.groupby('segment').agg(
        jumlah_customer=('customer_unique_id', 'count'),
        avg_frekuensi=('jumlah_order', 'mean'),
        avg_spending_per_order=('avg_spending', 'mean'),
    ).round(2)
    pivot['pct_customer'] = (
        pivot['jumlah_customer'] / pivot['jumlah_customer'].sum() * 100
    ).round(2)
    return pivot, frekuensi


# ── LOAD ──────────────────────────────────────────────────────
df_order_items, df_product, df_order_payments, df_order, df_customer, df_kategori, orders_delivered = load_data()
pivot_kategori = process_q1_base(df_order_items, df_product, df_kategori, orders_delivered)
pivot_revenue_base, payment_agg, df_revenue_base = process_q2_base(orders_delivered, df_order_payments)
pivot_customer, frekuensi = process_q3(orders_delivered, df_customer, payment_agg)

# ── SIDEBAR ───────────────────────────────────────────────────
with st.sidebar:
    st.title("🎛️ Filter Data")
    st.divider()

    # Filter Q1
    st.subheader("📦 Q1 — Kategori Produk")
    metrik_q1 = st.radio(
        "Tampilkan berdasarkan:",
        options=["total_items_sold", "total_revenue", "avg_price"],
        format_func=lambda x: {
            "total_items_sold": "Jumlah Item Terjual",
            "total_revenue": "Total Revenue",
            "avg_price": "Rata-rata Harga"
        }[x]
    )

    st.divider()

    # Filter Q2
    st.subheader("📈 Q2 — Tren Revenue")
    min_date = df_revenue_base['order_purchase_timestamp'].min().date()
    max_date = df_revenue_base['order_purchase_timestamp'].max().date()

    date_start = st.date_input(
        "Dari tanggal:",
        value=min_date,
        min_value=min_date,
        max_value=max_date
    )
    date_end = st.date_input(
        "Sampai tanggal:",
        value=max_date,
        min_value=min_date,
        max_value=max_date
    )

# ── TERAPKAN FILTER ───────────────────────────────────────────

# Q1
pivot_q1_sorted = pivot_kategori.sort_values(metrik_q1, ascending=False)
top5 = pivot_q1_sorted.head(5)
bot5 = pivot_q1_sorted.tail(5).sort_values(metrik_q1, ascending=True)

# Q2
if date_start <= date_end:
    df_revenue_filtered = df_revenue_base[
        (df_revenue_base['order_purchase_timestamp'].dt.date >= date_start) &
        (df_revenue_base['order_purchase_timestamp'].dt.date <= date_end)
    ].copy()

    pivot_revenue = df_revenue_filtered.groupby('year_month').agg(
        total_revenue=('total_payment', 'sum'),
        order_count=('order_id', 'count'),
        avg_order_value=('total_payment', 'mean')
    ).round(2)
else:
    pivot_revenue = pd.DataFrame()

# ── HEADER ────────────────────────────────────────────────────
st.title("🛒 Olist E-Commerce Dashboard")
st.caption("Brazilian E-Commerce Public Dataset | Sep 2016 – Aug 2018")
st.divider()

# ── METRIC SUMMARY ────────────────────────────────────────────
col1, col2, col3, col4 = st.columns(4)
col1.metric("Total Item Terjual", f"{pivot_kategori['total_items_sold'].sum():,}")
col2.metric("Total Revenue (2017)", f"R$ {pivot_revenue_base['total_revenue'].sum():,.0f}")
col3.metric("Kategori Aktif", f"{len(pivot_kategori)}")
col4.metric("Unique Customer", f"{len(frekuensi):,}")
st.divider()

# ── TABS ──────────────────────────────────────────────────────
tab1, tab2, tab3 = st.tabs(["📦 Q1 — Produk Terlaris", "📈 Q2 — Tren Revenue", "👥 Q3 — Customer"])

# ═══════════════════════════════════════════════════════════
# TAB 1 — Q1
# ═══════════════════════════════════════════════════════════
with tab1:
    st.subheader("Produk dengan penjualan terbaik dan terburuk periode Januari 2017 - Agustus 2018")

    metrik_label = {
        "total_items_sold": "Jumlah Item Terjual",
        "total_revenue": "Total Revenue",
        "avg_price": "Rata-rata Harga"
    }[metrik_q1]

    colors = ["#72BCD4", "#D3D3D3", "#D3D3D3", "#D3D3D3", "#D3D3D3"]
    fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(24, 6))

    sns.barplot(x=metrik_q1, y=top5.index,
                data=top5.reset_index(), palette=colors, ax=ax[0])
    ax[0].set_ylabel(None)
    ax[0].set_xlabel(None)
    ax[0].set_title("Best Performing Category", loc="center", fontsize=15)
    ax[0].tick_params(axis='y', labelsize=12)

    sns.barplot(x=metrik_q1, y=bot5.index,
                data=bot5.reset_index(), palette=colors, ax=ax[1])
    ax[1].set_ylabel(None)
    ax[1].set_xlabel(None)
    ax[1].invert_xaxis()
    ax[1].yaxis.set_label_position("right")
    ax[1].yaxis.tick_right()
    ax[1].set_title("Worst Performing Category", loc="center", fontsize=15)
    ax[1].tick_params(axis='y', labelsize=12)

    plt.suptitle(f"Best and Worst Performing Category by {metrik_label}", fontsize=20)
    plt.tight_layout()
    st.pyplot(fig)
    plt.close()

    col_t1, col_t2 = st.columns(2)
    with col_t1:
        st.markdown("**Top 5 Kategori**")
        st.dataframe(top5[['total_items_sold', 'total_revenue', 'avg_price', 'pct_share']],
                     use_container_width=True)
    with col_t2:
        st.markdown("**Bottom 5 Kategori**")
        st.dataframe(bot5[['total_items_sold', 'total_revenue', 'avg_price', 'pct_share']],
                     use_container_width=True)

# ═══════════════════════════════════════════════════════════
# TAB 2 — Q2
# ═══════════════════════════════════════════════════════════
with tab2:
    st.subheader("Pertumbuhan total revenue bulanan platform Olist selama periode tahun 2017")

    if date_start > date_end:
        st.warning("Tanggal awal tidak boleh lebih besar dari tanggal akhir.")
    elif len(pivot_revenue) == 0:
        st.warning("Tidak ada data pada rentang tanggal yang dipilih.")
    else:
        fig, ax1 = plt.subplots(figsize=(14, 6))

        ax1.bar(
            pivot_revenue.index.astype(str),
            pivot_revenue['order_count'],
            color='#D3D3D3', label='Order Count'
        )
        ax1.set_ylabel('Jumlah Order', color='gray')
        ax1.tick_params(axis='x', rotation=45)
        ax1.tick_params(axis='y', labelcolor='gray')

        ax2 = ax1.twinx()
        ax2.plot(
            pivot_revenue.index.astype(str),
            pivot_revenue['total_revenue'],
            marker='o', color='#72BCD4', linewidth=2, markersize=5, label='Total Revenue'
        )
        ax2.set_ylabel('Total Revenue (R$)', color='#72BCD4')
        ax2.tick_params(axis='y', labelcolor='#72BCD4')

        lines, labels = ax2.get_legend_handles_labels()
        bars, blabels = ax1.get_legend_handles_labels()
        ax1.legend(bars + lines, blabels + labels, loc='upper left')

        plt.title(f'Tren Revenue & Jumlah Order Bulanan Olist ({date_start} – {date_end})', fontsize=15)
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()

        st.markdown("**Pivot Table Revenue Bulanan**")
        st.dataframe(pivot_revenue, use_container_width=True)

# ═══════════════════════════════════════════════════════════
# TAB 3 — Q3
# ═══════════════════════════════════════════════════════════
with tab3:
    st.subheader("New customers vs returning customers periode 2016 - 2018")
    
    col_pie, col_bar = st.columns([1, 2])

    with col_pie:
        fig, ax = plt.subplots(figsize=(5, 5))
        ax.pie(
            pivot_customer['jumlah_customer'],
            labels=pivot_customer.index,
            autopct='%1.1f%%',
            colors=['#72BCD4', '#D3D3D3'],
            startangle=90
        )
        ax.set_title('Komposisi New vs Returning Customer', fontsize=12)
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()

    with col_bar:
        fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(10, 5))

        ax[0].bar(pivot_customer.index, pivot_customer['avg_spending_per_order'],
                  color=['#72BCD4', '#D3D3D3'])
        ax[0].set_title('Rata-rata Spending per Order', fontsize=12)
        ax[0].set_ylabel('R$')
        for i, v in enumerate(pivot_customer['avg_spending_per_order']):
            ax[0].text(i, v + 1, f'R$ {v:,.2f}', ha='center', fontsize=10)

        ax[1].bar(pivot_customer.index, pivot_customer['avg_frekuensi'],
                  color=['#72BCD4', '#D3D3D3'])
        ax[1].set_title('Rata-rata Frekuensi Belanja', fontsize=12)
        ax[1].set_ylabel('Jumlah Order')
        for i, v in enumerate(pivot_customer['avg_frekuensi']):
            ax[1].text(i, v + 0.02, f'{v:.2f}x', ha='center', fontsize=10)

        plt.suptitle('Perbandingan Perilaku New vs Returning Customer', fontsize=13)
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()

    st.markdown("**Pivot Table Segmentasi Customer**")
    st.dataframe(pivot_customer, use_container_width=True)

# ── FOOTER ────────────────────────────────────────────────────
st.divider()
st.caption("Dashboard Proyek Fundamental Analisis Data | Data: Brazilian E-Commerce Public Dataset (Kaggle)")
