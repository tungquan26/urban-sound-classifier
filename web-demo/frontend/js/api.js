const ApiService = {
    async classifyAudio(file) {
        const formData = new FormData();
        formData.append('file', file);
        try {

            const response = await fetch('http://localhost:8000/predict', {
                method: 'POST',
                body: formData
            });
            if (!response.ok) throw new Error("Lỗi API");
            return await response.json();
        } catch (error) {
            console.error("API Error:", error);
            alert("Không thể kết nối đến AI Server!");
            return null;
        }
    }
}
