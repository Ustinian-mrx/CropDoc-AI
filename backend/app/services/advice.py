def build_advice(prediction: dict) -> dict:
    crop = prediction["crop"]
    disease = prediction["disease"]
    is_healthy = prediction["is_healthy"]
    confidence = prediction["confidence"]

    if is_healthy:
        return {
            "title": f"{crop} 健康状态建议",
            "risk_level": "low",
            "symptoms": [
                "当前图片未识别出明显病害特征。",
                "仍建议结合叶片背面、植株整体长势和田间环境继续观察。",
            ],
            "actions": [
                "保持正常水肥管理。",
                "定期观察叶片颜色、斑点、卷曲和枯萎情况。",
                "如果后续出现异常症状，可重新拍摄清晰图片进行识别。",
            ],
            "prevention": [
                "保持种植区域通风透光。",
                "避免长期高湿环境。",
                "及时清理病残叶和杂草。",
            ],
            "disclaimer": "本建议由 AI 系统生成，仅供参考，不能替代当地农技人员或植保专家诊断。",
        }

    risk_level = "medium"

    if confidence >= 0.95:
        risk_level = "high"
    elif confidence < 0.75:
        risk_level = "uncertain"

    return {
        "title": f"{crop} - {disease} 防治建议",
        "risk_level": risk_level,
        "symptoms": [
            f"模型识别结果显示该叶片疑似存在 {disease}。",
            "建议结合叶片正反面、病斑扩散范围、植株整体状态进行进一步确认。",
            "若同一区域多株植物出现类似症状，应提高重视。",
        ],
        "actions": [
            "隔离或标记疑似病株，避免病害进一步扩散。",
            "清理严重受害叶片，并妥善处理病残体。",
            "根据当地农技指导选择合适药剂或生物防治方案。",
            "避免在高温、强光或大风条件下随意施药。",
        ],
        "prevention": [
            "保持合理株距，增强通风透光。",
            "避免过度浇水和长期叶面潮湿。",
            "定期巡田，发现早期病斑及时处理。",
            "轮作或清理残留病源，降低复发概率。",
        ],
        "disclaimer": "本建议由 AI 系统根据图像识别结果生成，仅供学习和辅助参考，实际防治请结合当地农技人员建议。",
    }
