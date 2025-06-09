import streamlit as st
import time
import numpy as np

st.sidebar.text("Side Bar Page")
data = np.random.randn(100,2)

st.sidebar.bar_chart(data=data)
st.sidebar.scatter_chart(data)

progress_bar = st.sidebar.progress(0)
status_text = st.sidebar.empty()
last_rows = np.random.randn(1,1)
chart = st.line_chart(last_rows)
bar = st.bar_chart(last_rows)

for i in range(1,101):
    new_rows = last_rows[-1, :] + np.random.randn(5,1).cumsum(axis = 0)
    status_text.text("%i%% Complete" % i)
    chart.add_rows(new_rows)
    bar.add_rows(new_rows)
    progress_bar.progress(i)
    last_rows = new_rows
    time.sleep(0.05)

st.write("First App")
#progress_bar.empty()
st.button("Re-run")