# MicroFragment Atlas Pro / 微塑料破碎科研建模平台

A polished bilingual Streamlit app for microplastic fragmentation modelling, interpretable random forest analysis, and raster prediction.  
一个用于微塑料破碎建模、随机森林解释分析与栅格空间预测的双语 Streamlit 科研应用。

## Highlights / 功能亮点
- Scientific, elegant dashboard styling / 科研风格、简洁大气的界面
- Random Forest modelling from CSV / 基于 CSV 的随机森林建模
- Optional SHAP analysis to control memory usage / SHAP 默认可选，减少内存占用
- Single-feature SHAP exploration on demand / 支持按变量单独查看 SHAP
- Repeated CV and spatial CV / 支持重复交叉验证与空间交叉验证
- Raster prediction and uncertainty export / 支持栅格预测与不确定性输出

## Deploy / 部署
```bash
pip install -r requirements.txt
streamlit run app.py
```

## Recommended workflow / 推荐使用流程
1. Upload the sampling CSV / 上传采样 CSV  
2. Select target and coordinates / 选择目标变量与坐标字段  
3. Run the modelling workflow / 运行建模流程  
4. Enable SHAP only if needed / 仅在需要时启用 SHAP  
5. Upload aligned TIFF predictors / 上传已经对齐的 TIFF 预测因子  
6. Export raster prediction outputs / 导出栅格预测结果  

## Notes / 说明
SHAP is disabled by default to reduce memory pressure in cloud deployment.  
SHAP 默认关闭，以减少云端部署时的内存压力。
