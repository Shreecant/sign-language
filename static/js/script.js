window.onload = () => {
  const video = document.getElementById("video");
  const canvas = document.getElementById("canvas");
  const ctx = canvas.getContext("2d");
  const result = document.getElementById("result");
  const socket = io();

  navigator.mediaDevices.getUserMedia({ video: true })
    .then(stream => video.srcObject = stream)
    .catch(err => console.error("Webcam error:", err));

  // Send a frame every 100ms
  setInterval(() => {
    if (video.readyState === video.HAVE_ENOUGH_DATA) {
      ctx.drawImage(video, 0, 0, canvas.width, canvas.height);
      const dataURL = canvas.toDataURL("image/jpeg", 0.5);
      socket.emit("frame", dataURL);
    }
  }, 100);

  // Receive result
  socket.on("result", (data) => {
    result.textContent = data.message;
  });
};
