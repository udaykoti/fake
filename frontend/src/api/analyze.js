const API_URL = import.meta.env.VITE_API_URL || "http://localhost:8000";

export async function analyzeJob({ text, url, companyName, image }) {
  const form = new FormData();
  if (text)        form.append("text", text);
  if (url)         form.append("url", url);
  if (companyName) form.append("company_name", companyName);
  if (image)       form.append("image", image);

  const res = await fetch(`${API_URL}/analyze`, { method: "POST", body: form });
  if (!res.ok) {
    const err = await res.json().catch(() => ({}));
    throw new Error(err.detail || `Request failed: ${res.status}`);
  }
  return res.json();
}
