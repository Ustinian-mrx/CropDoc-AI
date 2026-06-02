"use client";

import { ChangeEvent, useMemo, useState } from "react";

type PredictionResult = {
  class_name: string;
  crop: string;
  disease: string;
  is_healthy: boolean;
  confidence: number;
};

type PredictResponse = {
  filename: string;
  model: {
    name: string;
    version: string;
    image_size: number;
  };
  prediction: PredictionResult;
  top3: PredictionResult[];
  explanation: {
    gradcam_available: boolean;
    heatmap_url: string | null;
  };
  note: string;
  advice: Advice;
};

type Advice = {
  title: string;
  risk_level: string;
  symptoms: string[];
  actions: string[];
  prevention: string[];
  disclaimer: string;
};

type HistoryRecord = {
  id: number;
  filename: string;
  prediction: {
    crop: string;
    disease: string;
    is_healthy: boolean;
    confidence: number;
  };
  top3: PredictionResult[];
  advice: Advice;
  explanation: {
    gradcam_available: boolean;
    heatmap_url: string | null;
  };
  created_at: string;
};

type HistoryResponse = {
  items: HistoryRecord[];
  count: number;
};


export default function Home() {
  const [selectedFile, setSelectedFile] = useState<File | null>(null);
  const [previewUrl, setPreviewUrl] = useState<string | null>(null);
  const [result, setResult] = useState<PredictResponse | null>(null);
  const [historyItems, setHistoryItems] = useState<HistoryRecord[]>([]);
  const [isHistoryLoading, setIsHistoryLoading] = useState(false);
  const [historyError, setHistoryError] = useState<string | null>(null);
  const [isLoading, setIsLoading] = useState(false);
  const [errorMessage, setErrorMessage] = useState<string | null>(null);

  const apiBaseUrl = useMemo(() => {
    return process.env.NEXT_PUBLIC_API_BASE_URL ?? "http://127.0.0.1:8000";
  }, []);

  function handleFileChange(event: ChangeEvent<HTMLInputElement>) {
    const file = event.target.files?.[0];

    if (!file) {
      return;
    }

    setSelectedFile(file);
    setPreviewUrl(URL.createObjectURL(file));
    setResult(null);
    setErrorMessage(null);
  }

  async function handlePredict() {
    if (!selectedFile) {
      setErrorMessage("请先选择一张叶片图片。");
      return;
    }

    setIsLoading(true);
    setErrorMessage(null);
    setResult(null);

    try {
      const formData = new FormData();
      formData.append("file", selectedFile);

      const response = await fetch(`${apiBaseUrl}/predict`, {
        method: "POST",
        body: formData,
      });

      if (!response.ok) {
        const errorData = await response.json().catch(() => null);
        throw new Error(errorData?.detail ?? "诊断请求失败。");
      }

      const data = (await response.json()) as PredictResponse;
      setResult(data);
      await loadHistory();
    } catch (error) {
      const message = error instanceof Error ? error.message : "未知错误。";
      setErrorMessage(message);
    } finally {
      setIsLoading(false);
    }
  }

  async function loadHistory() {
    setIsHistoryLoading(true);
    setHistoryError(null);

    try {
      const response = await fetch(`${apiBaseUrl}/history?limit=10`);

      if (!response.ok) {
        throw new Error("历史记录加载失败。");
      }

      const data = (await response.json()) as HistoryResponse;
      setHistoryItems(data.items);
    } catch (error) {
      const message = error instanceof Error ? error.message : "未知错误。";
      setHistoryError(message);
    } finally {
      setIsHistoryLoading(false);
    }
  }


  function formatPercent(value: number) {
    return `${(value * 100).toFixed(2)}%`;
  }

  return (
    <main className="min-h-screen bg-slate-950 text-slate-100">
      <div className="mx-auto flex min-h-screen w-full max-w-6xl flex-col gap-8 px-6 py-8">
        <header className="border-b border-slate-800 pb-6">
          <p className="text-sm font-medium uppercase tracking-wide text-emerald-400">
            CropDoc AI
          </p>
          <h1 className="mt-3 text-3xl font-semibold text-white">
            农作物叶片病害识别
          </h1>
          <p className="mt-3 max-w-2xl text-sm leading-6 text-slate-300">
            上传叶片图片，系统会返回识别结果、置信度和 Top-3 候选类别。
          </p>
        </header>

        <section className="grid gap-6 lg:grid-cols-[1fr_1fr]">
          <div className="rounded-lg border border-slate-800 bg-slate-900 p-5">
            <h2 className="text-lg font-medium text-white">上传图片</h2>

            <div className="mt-4 flex flex-col gap-4">
              <input
                type="file"
                accept="image/*"
                onChange={handleFileChange}
                className="block w-full text-sm text-slate-300 file:mr-4 file:rounded-md file:border-0 file:bg-emerald-500 file:px-4 file:py-2 file:text-sm file:font-medium file:text-slate-950 hover:file:bg-emerald-400"
              />

              {previewUrl ? (
                <div className="overflow-hidden rounded-md border border-slate-800 bg-slate-950">
                  <img
                    src={previewUrl}
                    alt="Selected leaf"
                    className="h-80 w-full object-contain"
                  />
                </div>
              ) : (
                <div className="flex h-80 items-center justify-center rounded-md border border-dashed border-slate-700 bg-slate-950 text-sm text-slate-500">
                  请选择一张图片
                </div>
              )}

              <button
                type="button"
                onClick={handlePredict}
                disabled={isLoading}
                className="rounded-md bg-emerald-500 px-4 py-3 text-sm font-semibold text-slate-950 transition hover:bg-emerald-400 disabled:cursor-not-allowed disabled:bg-slate-700 disabled:text-slate-400"
              >
                {isLoading ? "诊断中..." : "开始诊断"}
              </button>

              {errorMessage ? (
                <p className="rounded-md border border-red-900 bg-red-950 px-4 py-3 text-sm text-red-200">
                  {errorMessage}
                </p>
              ) : null}
            </div>
          </div>

          <div className="rounded-lg border border-slate-800 bg-slate-900 p-5">
            <h2 className="text-lg font-medium text-white">诊断结果</h2>

            {result ? (
              <div className="mt-4 flex flex-col gap-5">
                <div className="rounded-md border border-slate-800 bg-slate-950 p-4">
                  <p className="text-sm text-slate-400">主要判断</p>
                  <p className="mt-2 text-2xl font-semibold text-white">
                    {result.prediction.disease}
                  </p>
                  <p className="mt-1 text-sm text-slate-300">
                    作物：{result.prediction.crop}
                  </p>
                  <p className="mt-1 text-sm text-slate-300">
                    状态：
                    {result.prediction.is_healthy ? "健康" : "疑似病害"}
                  </p>
                  <p className="mt-3 text-sm font-medium text-emerald-300">
                    置信度：{formatPercent(result.prediction.confidence)}
                  </p>
                  <p className="mt-3 text-xs leading-5 text-slate-500">
                    {result.note}
                  </p>

                  <p className="mt-2 text-xs text-slate-500">
                    Model: {result.model.name} / v{result.model.version}
                  </p>

                </div>
                {result.explanation.gradcam_available && result.explanation.heatmap_url ? (
                  <div className="rounded-md border border-slate-800 bg-slate-950 p-4">
                    <p className="text-sm font-medium text-slate-300">模型关注区域</p>
                    <img
                      src={`${apiBaseUrl}${result.explanation.heatmap_url}`}
                      alt="Grad-CAM heatmap"
                      className="mt-3 h-80 w-full rounded-md object-contain"
                    />
                  </div>
                ) : null}
                <div className="rounded-md border border-slate-800 bg-slate-950 p-4">
                  <div className="flex items-start justify-between gap-4">
                    <div>
                      <p className="text-sm font-medium text-slate-300">防治建议</p>
                      <h3 className="mt-2 text-lg font-semibold text-white">
                        {result.advice.title}
                      </h3>
                    </div>
                    <span className="rounded-md bg-emerald-500 px-3 py-1 text-xs font-semibold text-slate-950">
                      {result.advice.risk_level}
                    </span>
                  </div>

                  <div className="mt-4 grid gap-4 md:grid-cols-3">
                    <div>
                      <p className="text-sm font-medium text-slate-300">症状参考</p>
                      <ul className="mt-2 space-y-2 text-sm leading-6 text-slate-400">
                        {result.advice.symptoms.map((item) => (
                          <li key={item}>{item}</li>
                        ))}
                      </ul>
                    </div>

                    <div>
                      <p className="text-sm font-medium text-slate-300">处理措施</p>
                      <ul className="mt-2 space-y-2 text-sm leading-6 text-slate-400">
                        {result.advice.actions.map((item) => (
                          <li key={item}>{item}</li>
                        ))}
                      </ul>
                    </div>

                    <div>
                      <p className="text-sm font-medium text-slate-300">预防建议</p>
                      <ul className="mt-2 space-y-2 text-sm leading-6 text-slate-400">
                        {result.advice.prevention.map((item) => (
                          <li key={item}>{item}</li>
                        ))}
                      </ul>
                    </div>
                  </div>

                  <p className="mt-4 text-xs leading-5 text-slate-500">
                    {result.advice.disclaimer}
                  </p>
                </div>

                <div>
                  <h3 className="text-sm font-medium text-slate-300">
                    Top-3 候选结果
                  </h3>

                  <div className="mt-3 flex flex-col gap-3">
                    {result.top3.map((item, index) => (
                      <div
                        key={`${item.class_name}-${index}`}
                        className="rounded-md border border-slate-800 bg-slate-950 p-4"
                      >
                        <div className="flex items-center justify-between gap-4">
                          <div>
                            <p className="text-sm font-medium text-white">
                              {index + 1}. {item.disease}
                            </p>
                            <p className="mt-1 text-xs text-slate-400">
                              {item.crop} / {item.class_name}
                            </p>
                          </div>

                          <p className="text-sm font-semibold text-emerald-300">
                            {formatPercent(item.confidence)}
                          </p>
                        </div>

                        <div className="mt-3 h-2 overflow-hidden rounded-full bg-slate-800">
                          <div
                            className="h-full rounded-full bg-emerald-400"
                            style={{
                              width: `${Math.max(item.confidence * 100, 2)}%`,
                            }}
                          />
                        </div>
                      </div>
                    ))}
                  </div>
                </div>
              </div>
            ) : (
              <div className="mt-4 flex h-80 items-center justify-center rounded-md border border-dashed border-slate-700 bg-slate-950 text-sm text-slate-500">
                诊断结果会显示在这里
              </div>
            )}
          </div>
        </section>
        <section className="rounded-lg border border-slate-800 bg-slate-900 p-5">
          <div className="flex items-center justify-between gap-4">
            <div>
              <h2 className="text-lg font-medium text-white">历史记录</h2>
              <p className="mt-1 text-sm text-slate-400">
                查看最近 10 次诊断结果。
              </p>
            </div>

            <button
              type="button"
              onClick={loadHistory}
              disabled={isHistoryLoading}
              className="rounded-md border border-slate-700 px-4 py-2 text-sm font-medium text-slate-200 transition hover:border-emerald-400 hover:text-emerald-300 disabled:cursor-not-allowed disabled:text-slate-500"
            >
              {isHistoryLoading ? "加载中..." : "刷新"}
            </button>
          </div>

          {historyError ? (
            <p className="mt-4 rounded-md border border-red-900 bg-red-950 px-4 py-3 text-sm text-red-200">
              {historyError}
            </p>
          ) : null}

          <div className="mt-4 grid gap-3">
            {historyItems.length > 0 ? (
              historyItems.map((item) => (
                <div
                  key={item.id}
                  className="rounded-md border border-slate-800 bg-slate-950 p-4"
                >
                  <div className="flex flex-col justify-between gap-3 md:flex-row md:items-center">
                    <div>
                      <p className="text-sm font-medium text-white">
                        {item.prediction.crop} / {item.prediction.disease}
                      </p>
                      <p className="mt-1 text-xs text-slate-500">
                        {item.filename} · {item.created_at}
                      </p>
                    </div>

                    <div className="flex items-center gap-3">
                      <span className="rounded-md bg-slate-800 px-3 py-1 text-xs text-slate-300">
                        {item.prediction.is_healthy ? "健康" : "疑似病害"}
                      </span>

                      <span className="text-sm font-semibold text-emerald-300">
                        {formatPercent(item.prediction.confidence)}
                      </span>

                      {item.explanation.heatmap_url ? (
                        <a
                          href={`${apiBaseUrl}${item.explanation.heatmap_url}`}
                          target="_blank"
                          rel="noreferrer"
                          className="text-sm text-emerald-300 hover:text-emerald-200"
                        >
                          热力图
                        </a>
                      ) : null}
                    </div>
                  </div>
                </div>
              ))
            ) : (
              <div className="rounded-md border border-dashed border-slate-700 bg-slate-950 p-6 text-center text-sm text-slate-500">
                暂无历史记录
              </div>
            )}
          </div>
        </section>
      </div>
    </main>
  );
}
