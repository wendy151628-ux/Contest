#%%
import pandas as pd
df2020 = pd.read_csv("C:/Users/jeongmin/Downloads/vscode/2020점포.csv", encoding='cp949')
# 행정동 코드가 '1150'으로 시작하는 데이터 필터링
df2020 = df2020[df2020["행정동_코드"].astype(str).str.startswith("1150")]
# 분기별 + 행정동코드별 점포수 합계 계산
df2020 = df2020.groupby(["기준_년분기_코드", "행정동_코드"], as_index=False)["점포_수"].sum()

# %%
df2021 = pd.read_csv("C:/Users/jeongmin/Downloads/vscode/2021점포.csv", encoding='cp949')
df2021 = df2021[df2021["행정동_코드"].astype(str).str.startswith("1150")]
df2021 = df2021.groupby(["기준_년분기_코드", "행정동_코드"], as_index=False)["점포_수"].sum()
df2022 = pd.read_csv("C:/Users/jeongmin/Downloads/vscode/2022점포.csv", encoding='cp949')
df2022 = df2022[df2022["행정동_코드"].astype(str).str.startswith("1150")]
df2022 = df2022.groupby(["기준_년분기_코드", "행정동_코드"], as_index=False)["점포_수"].sum()
df2023 = pd.read_csv("C:/Users/jeongmin/Downloads/vscode/2023점포.csv", encoding='cp949')
df2023 = df2023[df2023["행정동_코드"].astype(str).str.startswith("1150")]
df2023 = df2023.groupby(["기준_년분기_코드", "행정동_코드"], as_index=False)["점포_수"].sum()
df2024 = pd.read_csv("C:/Users/jeongmin/Downloads/vscode/2024점포.csv", encoding='cp949')
df2024 = df2024[df2024["행정동_코드"].astype(str).str.startswith("1150")]
df2024 = df2024.groupby(["기준_년분기_코드", "행정동_코드"], as_index=False)["점포_수"].sum()
# %%
df_all = pd.concat([df2020, df2021, df2022, df2023, df2024], ignore_index=True)
df_all = df_all.sort_values(by=["기준_년분기_코드", "행정동_코드"])
# %%
df_all.to_csv("강서구점포수.csv", encoding='cp949')



# %% 필요한 패키지 모음 및 그래프 깨짐 방지
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
import numpy as np
from tslearn.metrics import cdist_dtw
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import silhouette_score
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import MinMaxScaler
from statsmodels.nonparametric.smoothers_lowess import lowess
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KernelDensity
from sklearn.neighbors import NearestNeighbors
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
import statsmodels.api as sm
plt.rcParams['font.family'] = 'Malgun Gothic'
plt.rcParams['axes.unicode_minus'] = False


# %% 강서구 데이터 모음 읽기
gangseo = pd.read_csv("C:/Users/jeongmin/Downloads/vscode/강서구.csv", encoding='utf-8')

# %% 동별면적(제곱km단위) 이용한 밀도 계산
area_df = pd.DataFrame({
    "행정동명": ["염창동","등촌1동","등촌2동","등촌3동","화곡본동","화곡2동","화곡3동","화곡4동","화곡6동","화곡8동",
             "가양1동","가양2동","가양3동","발산1동","공항동","방화1동","방화2동","방화3동","화곡1동","우장산동"],
    "면적": [1.74,0.64,0.92,0.79,0.98,0.45,0.53,0.82,1.11,0.53,
           4.7,1,0.5,2.94,10.87,1.48,6.41,2.55,1.12,1.36]
})

gangseo = gangseo.merge(area_df, on="행정동명", how="left")
gangseo["유동인구_밀도"] = gangseo["총_유동인구_수"] / gangseo["면적"]
for col in ["상주인구", "평일유동인구", "주말유동인구", "집객시설", "버스정거장수", "점포수"]:
    gangseo[f"{col}_밀도"] = gangseo[col] / gangseo["면적"]
final = gangseo.drop(columns=["면적", "점포수", "상주인구", "총_유동인구_수", "평일유동인구", "주말유동인구", "집객시설", "버스정거장수"])

final




# %% 피벗: 행=행정동명, 열=date, 값=지출총금액
final["date"] = final["date"].astype(int)
pivot = final.pivot_table(index="행정동명", columns="date", values="지출총금액", aggfunc="sum")
pivot = pivot.sort_index(axis=1)
ts = pivot.dropna(axis=0)

# %% DTW 거리행렬 & 최적 k 탐색 (변수 min-max scaling)
ts = pivot.dropna(axis=0)
scaler = MinMaxScaler()
ts_norm = ts.apply(lambda row: scaler.fit_transform(row.values.reshape(-1,1)).ravel(), axis=1, result_type="expand")
ts_norm.columns = ts.columns
X = ts_norm.to_numpy()
D = cdist_dtw(X)

def choose_best_k(distance_matrix, k_range=range(2, 9)):
    best_k, best_score, best_labels = None, -1, None
    for k in k_range:
        model = AgglomerativeClustering(
            n_clusters=k, metric="precomputed", linkage="average"
        )
        labels = model.fit_predict(distance_matrix)
        score = silhouette_score(distance_matrix, labels, metric="precomputed")
        print(f"k={k}, silhouette={score:.4f}")
        if score > best_score:
            best_k, best_score, best_labels = k, score, labels
    return best_k, best_score, best_labels

best_k, best_sil, cluster_labels = choose_best_k(D, k_range=range(2, 9))
print("Best k:", best_k, "Silhouette:", best_sil)


# %% 최종 군집 적합
final_model = AgglomerativeClustering(
    n_clusters=2, metric="precomputed", linkage="average"
)
cluster_labels = final_model.fit_predict(D)

# 결과 저장
cluster_result = pd.DataFrame({
    "행정동명": ts_norm.index,
    "cluster": cluster_labels
})
print(cluster_result.sort_values("cluster"))


# %% 그룹별 선형 회귀 해보기 .. plot은 안그렸음
dates = ts_norm.columns.to_list()
x_vals = np.arange(len(dates))  # 분기 index
for lab in sorted(set(cluster_labels)):
    cluster_series = ts_norm.iloc[[i for i, lbl in enumerate(cluster_labels) if lbl == lab]]
    # 군집 전체를 하나의 데이터셋으로 풀기 (x는 분기 index 반복, y는 값들)
    xs, ys = [], []
    for row in cluster_series.values:
        xs.extend(list(range(len(dates))))
        ys.extend(row.tolist())
    # 회귀 모델 적합
    X = sm.add_constant(xs)  # 절편 추가
    model = sm.OLS(ys, X).fit()
    print(f"📊 Cluster {lab} 회귀 요약")
    print(model.summary())


# %% 시각화 (점 크기=근처 이웃 수, 비모수추세선=LOWESS)
for lab in sorted(set(cluster_labels)):
    cluster_series = ts_norm.iloc[[i for i, lbl in enumerate(cluster_labels) if lbl == lab]]

    date_labels = [f"{str(d)[:4]}_{str(d)[-1]}" for d in dates]
    x_vals = np.arange(len(dates)).reshape(-1,1)  # 회귀용 x축 (숫자)
    #모든 점 모으기
    xs, ys = [], []
    for row in cluster_series.values:
        xs.extend(list(range(len(dates))))
        ys.extend(row.tolist())
    X_points = np.vstack([xs, ys]).T

    # 반경 r 내 이웃 개수 = 점 크기
    r = 0.05  # y축 값 기준 (정규화 했으니 0~1 범위면 0.05~0.1 적당)
    nbrs = NearestNeighbors(radius=r).fit(X_points)
    sizes = np.array([
        len(nbrs.radius_neighbors([pt], return_distance=False)[0]) 
        for pt in X_points
    ]) * 100  # 배율 조정

    # LOWESS 비모수 추세선
    # xs는 분기 인덱스, ys는 값
    smooth = lowess(ys, xs, frac=0.7)  # frac=0.3: 스무딩 정도 (0.1~0.5 조정 가능)
    x_smooth, y_smooth = smooth[:,0], smooth[:,1]

    # 시각화
    plt.figure(figsize=(12,6))
    plt.scatter([date_labels[x] for x in xs], ys, s=sizes,
                alpha=0.4, color="skyblue", label="군집 시계열 점 (크기=밀집도)")
    plt.plot([date_labels[int(x)] for x in x_smooth], y_smooth,
             color="red", linewidth=3, label="LOWESS 추세선")
    plt.title(f"Cluster {lab}")
    plt.xlabel("date (분기)")
    plt.ylabel("지출총금액 (정규화 값)")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()


# %% 로지스틱 회귀
from statsmodels.stats.outliers_influence import variance_inflation_factor

data = final.copy()
data = data.merge(cluster_result, on="행정동명")
y = data["cluster"]
X = data.drop(columns=["평일유동인구_밀도", "주말유동인구_밀도", "상권변화", "지출총금액", "행정동코드", "행정동명", "cluster", "date"])

# VIF 계산
def calculate_vif(X):
    vif_data = pd.DataFrame()
    vif_data["feature"] = X.columns
    vif_data["VIF"] = [variance_inflation_factor(X.values, i)
                       for i in range(X.shape[1])]
    return vif_data

vif_table = calculate_vif(X)
print("초기 VIF:\n", vif_table)
selected_features = vif_table[vif_table["VIF"] < 10]["feature"].tolist()
X_reduced = X[selected_features]

# 로지스틱 회귀 (statsmodels)
X_reduced = sm.add_constant(X_reduced)
logit_model = sm.Logit(y, X_reduced)
result = logit_model.fit()
summary_table = pd.DataFrame({
    "Estimate": result.params,
    "StdErr": result.bse,
    "z value": result.tvalues,
    "p value": result.pvalues,
    "Odds Ratio": np.exp(result.params)
})
print(summary_table)

# %%
import geopandas as gpd
from libpysal.weights import Queen
from esda.moran import Moran
import spreg

gdf = gpd.read_file("C:/Users/jeongmin/Downloads/vscode/서울시 상권분석서비스(영역-행정동)/서울시 상권분석서비스(영역-행정동).shp", encoding="utf-8")
df = pd.read_excel("C:/Users/jeongmin/Downloads/vscode/강서구상권.xlsx")
data2 = gdf.merge(df, left_on="ADSTRD_NM", right_on="행정동명")
data = data.merge(data2, on="행정동명")

y = data["cluster"].values
X = data[["생활용품지출", "교육지출", "지하철역_더미", "유동인구_밀도"]].values

# 공간 가중치 행렬 (Queen contiguity: 인접 행정동)
w = Queen.from_dataframe(data)
w.transform = "r"

# Moran’s I (종속변수의 공간적 자기상관 확인)
moran = Moran(y, w)
print("Moran’s I:", moran.I, "p-value:", moran.p_sim)

# 공간 회귀 모형 (Spatial Lag)
model_lag = spreg.ML_Lag(y, X, w=w, name_y="cluster", 
                         name_x=["생활용품지출", "교육지출", "지하철역_더미", "유동인구_밀도"])
print(model_lag.summary)

# 예측값을 GeoDataFrame에 추가
data["pred_cluster"] = model_lag.predy.flatten()

#%%
from shapely.geometry import Point
# data 자체가 GeoDataFrame인지 보정
data = gpd.GeoDataFrame(data, geometry="geometry", crs=gdf.crs)

# threshold 기준으로 high_prob_areas 추출
threshold = data["pred_cluster"].quantile(0.95)
high_prob_areas = data[data["pred_cluster"] >= threshold].copy()

# 다시 GeoDataFrame으로 변환 (geometry 유지)
high_prob_areas = gpd.GeoDataFrame(high_prob_areas, geometry="geometry", crs=gdf.crs)

# buffer (500m)
high_prob_buffer = high_prob_areas.buffer(300)

# 상권점도 GeoDataFrame으로 변환 (좌표계 동일하게)
df["geometry"] = df.apply(lambda row: Point(row["엑스좌표_값"], row["와이좌표_값"]), axis=1)
store_gdf = gpd.GeoDataFrame(df, geometry="geometry", crs=gdf.crs)

# buffer 내부의 상권만 추출
selected_stores = store_gdf[store_gdf.within(high_prob_buffer.union_all())]

# %%
data_unique = data.drop_duplicates(subset=["행정동명"]).reset_index(drop=True)
fig, ax = plt.subplots(figsize=(10,8))
data_unique.plot(
    column="pred_cluster",
    cmap="RdYlGn_r",
    legend=True,
    ax=ax,
)
plt.title("상권 고변동 확률 지도", fontsize=14)
plt.axis("off")
plt.show()


# -------------------------------
# 지도 2: 최종 후보 상권 지도
# -------------------------------
fig, ax = plt.subplots(figsize=(10,8))
# 배경 (모든 행정동 회색)
data.plot(ax=ax, color="lightgrey", edgecolor="black")
# 고확률 지역 (오렌지)
high_prob_areas.plot(ax=ax, color="orange", alpha=0.5, edgecolor="red", label="고확률 지역")
# 선정된 상권 (파란 점)
selected_stores.plot(ax=ax, color="blue", markersize=30, label="선정된 상권")

plt.legend()
plt.title("최종 선정된 상권 후보", fontsize=14)
plt.axis("off")
plt.show()


# %%
