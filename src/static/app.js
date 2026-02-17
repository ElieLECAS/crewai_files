// JavaScript pour interactions supplémentaires

document.addEventListener("DOMContentLoaded", function () {
    // Ajouter des fonctionnalités de tri aux tableaux si nécessaire
    const tables = document.querySelectorAll(".data-table");

    tables.forEach((table) => {
        const headers = table.querySelectorAll("th");
        headers.forEach((header, index) => {
            header.style.cursor = "pointer";
            header.addEventListener("click", () => {
                sortTable(table, index);
            });
        });
    });

    // Initialiser le système de tâches s'il existe sur la page
    initUploadSystem();
});

function initUploadSystem() {
    const uploadArea = document.getElementById("upload-area");
    if (!uploadArea) return;

    const fileInput = document.getElementById("file-input");
    const uploadForm = document.getElementById("upload-form");
    const uploadStatus = document.getElementById("upload-status");
    const uploadBtn = document.getElementById("upload-btn");
    const fileList = document.getElementById("file-list");
    const tasksContainer = document.getElementById("tasks-container");
    const tasksList = document.getElementById("tasks-list");
    const tasksCount = document.getElementById("tasks-count");

    let pollingInterval = null;
    let isProcessing = false;

    // Click sur la zone d'upload
    uploadArea.addEventListener("click", () => fileInput.click());

    // Drag & drop
    uploadArea.addEventListener("dragover", (e) => {
        e.preventDefault();
        uploadArea.classList.add("dragover");
    });

    uploadArea.addEventListener("dragleave", () => {
        uploadArea.classList.remove("dragover");
    });

    uploadArea.addEventListener("drop", (e) => {
        e.preventDefault();
        uploadArea.classList.remove("dragover");
        const files = Array.from(e.dataTransfer.files).filter(
            (f) => f.type === "application/pdf",
        );
        if (files.length > 0) {
            const dataTransfer = new DataTransfer();
            files.forEach((file) => dataTransfer.items.add(file));
            fileInput.files = dataTransfer.files;
            updateFileList(files);
        }
    });

    fileInput.addEventListener("change", (e) => {
        if (e.target.files.length > 0) {
            updateFileList(Array.from(e.target.files));
        }
    });

    function updateFileList(files) {
        const hint = uploadArea.querySelector(".upload-hint");
        if (files.length === 0) {
            hint.textContent = "ou cliquez pour sélectionner";
            fileList.innerHTML = "";
        } else if (files.length === 1) {
            hint.textContent = `Fichier sélectionné : ${files[0].name}`;
            fileList.innerHTML = "";
        } else {
            hint.textContent = `${files.length} fichiers sélectionnés`;
            fileList.innerHTML =
                '<div class="file-list-header">Fichiers sélectionnés :</div>' +
                files
                    .map(
                        (file, index) =>
                            `<div class="file-item">
                        <span class="file-name">${file.name}</span>
                        <span class="file-size">(${(file.size / 1024).toFixed(1)} KB)</span>
                    </div>`,
                    )
                    .join("");
        }
    }

    // Soumission du formulaire
    uploadForm.addEventListener("submit", async (e) => {
        e.preventDefault();

        if (!fileInput.files.length) {
            showStatus(
                "Veuillez sélectionner au moins un fichier PDF",
                "error",
            );
            return;
        }

        const files = Array.from(fileInput.files);
        const formData = new FormData();
        files.forEach((file) => {
            formData.append("files", file);
        });

        uploadBtn.disabled = true;
        uploadBtn.innerHTML = `<span>Envoi de ${files.length} fichier(s)...</span>`;
        showStatus(`Envoi de ${files.length} fichier(s) en cours...`, "info");

        try {
            const response = await fetch("/upload", {
                method: "POST",
                body: formData,
            });

            const data = await response.json();

            if (data.success) {
                showStatus(
                    `✅ ${data.message}. Suivez la progression ci-dessous.`,
                    "success",
                );
                fileInput.value = "";
                updateFileList([]);

                // Afficher le conteneur de tâches et lancer le polling
                tasksContainer.style.display = "block";
                startPolling();
            } else {
                showStatus(`❌ ${data.message}`, "error");
            }
        } catch (error) {
            showStatus(`❌ Erreur : ${error.message}`, "error");
        } finally {
            uploadBtn.disabled = false;
            uploadBtn.innerHTML = "<span>Traiter les documents</span>";
        }
    });

    function showStatus(message, type) {
        uploadStatus.textContent = message;
        uploadStatus.className = `upload-status ${type}`;
        uploadStatus.style.display = "block";
    }

    async function fetchTaskStatus() {
        try {
            const response = await fetch("/api/upload/status");
            const tasks = await response.json();
            updateTasksUI(tasks);

            // Vérifier s'il reste des tâches en cours
            const activeTasks = tasks.filter(
                (t) => t.status === "processing" || t.status === "pending",
            );
            tasksCount.textContent = `${activeTasks.length} en cours`;

            if (activeTasks.length === 0 && isProcessing) {
                // Tout est fini, on rafraîchit les stats après un court délai
                isProcessing = false;
                setTimeout(() => {
                    refreshStats();
                }, 1000);
                // Arrêter le polling quand il n'y a plus de tâches actives
                stopPolling();
            } else if (activeTasks.length > 0) {
                isProcessing = true;
                tasksContainer.style.display = "block";
            } else if (activeTasks.length === 0 && !isProcessing) {
                // Pas de tâches actives et pas en cours de traitement, arrêter le polling
                stopPolling();
            }
            
            // Retourner le nombre de tâches actives pour la logique de démarrage
            return activeTasks.length;
        } catch (error) {
            console.error("Erreur lors de la récupération des statuts:", error);
            return 0;
        }
    }

    function updateTasksUI(tasks) {
        if (tasks.length === 0) {
            tasksList.innerHTML =
                '<div class="empty-state"><p>Aucun traitement récent</p></div>';
            return;
        }

        tasksList.innerHTML = tasks
            .map((task) => {
                let iconClass = "";
                let icon = "";

                switch (task.status) {
                    case "completed":
                        icon =
                            '<svg class="success" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M5 13l4 4L19 7"></path></svg>';
                        break;
                    case "failed":
                        icon =
                            '<svg class="error" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M6 18L18 6M6 6l12 12"></path></svg>';
                        break;
                    case "processing":
                        icon =
                            '<svg class="primary spin" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M4 4v5h.582m15.356 2A8.001 8.001 0 004.582 9m0 0H9m11 11v-5h-.581m0 0a8.003 8.003 0 01-15.357-2m15.357 2H15"></path></svg>';
                        break;
                    default:
                        icon =
                            '<svg class="secondary" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M12 8v4l3 3m6-3a9 9 0 11-18 0 9 9 0 0118 0z"></path></svg>';
                }

                const time = new Date(task.timestamp).toLocaleTimeString();

                return `
                <div class="task-item ${task.status}">
                    <div class="task-icon">${icon}</div>
                    <div class="task-info">
                        <span class="task-file">${task.file_name}</span>
                        <div class="task-message">${task.message}</div>
                    </div>
                    <div class="task-time">${time}</div>
                </div>
            `;
            })
            .join("");
    }

    function startPolling() {
        if (pollingInterval) return;
        fetchTaskStatus();
        pollingInterval = setInterval(fetchTaskStatus, 5000); // Polling toutes les 5 secondes
    }

    function stopPolling() {
        if (pollingInterval) {
            clearInterval(pollingInterval);
            pollingInterval = null;
        }
    }

    async function refreshStats() {
        try {
            const response = await fetch("/api/collections");
            const stats = await response.json();

            // Mettre à jour les nombres sur les cartes de stats
            for (const [name, data] of Object.entries(stats)) {
                const card = document.querySelector(
                    `.stat-card h3:contains("${data.display_name}")`,
                );
                if (card) {
                    const numberEl =
                        card.parentElement.querySelector(".stat-number");
                    if (numberEl) numberEl.textContent = data.count;
                }
            }

            // Fallback : recharger si on ne trouve pas les éléments
            // (Le sélecteur :contains n'est pas standard CSS, on va faire plus simple)
            document.querySelectorAll(".stat-card").forEach((card) => {
                const title = card
                    .querySelector("h3")
                    .textContent.toLowerCase();
                for (const [name, data] of Object.entries(stats)) {
                    if (title === data.display_name.toLowerCase()) {
                        card.querySelector(".stat-number").textContent =
                            data.count;
                    }
                }
            });

            showStatus(
                "✅ Statistiques mises à jour automatiquement",
                "success",
            );
        } catch (error) {
            console.error("Erreur rafraîchissement stats:", error);
        }
    }

    // Charger les statuts une seule fois au démarrage pour vérifier s'il y a des tâches actives
    fetchTaskStatus().then((activeCount) => {
        // Si des tâches sont actives, démarrer le polling
        if (activeCount > 0) {
            startPolling();
        }
    });
}

function sortTable(table, columnIndex) {
    const tbody = table.querySelector("tbody");
    const rows = Array.from(tbody.querySelectorAll("tr"));

    const isAscending = table.dataset.sortDirection !== "asc";
    table.dataset.sortDirection = isAscending ? "asc" : "desc";

    rows.sort((a, b) => {
        const aText = a.cells[columnIndex].textContent.trim();
        const bText = b.cells[columnIndex].textContent.trim();

        // Essayer de comparer comme des nombres
        const aNum = parseFloat(aText.replace(/[^\d.-]/g, ""));
        const bNum = parseFloat(bText.replace(/[^\d.-]/g, ""));

        if (!isNaN(aNum) && !isNaN(bNum)) {
            return isAscending ? aNum - bNum : bNum - aNum;
        }

        // Sinon comparer comme du texte
        return isAscending
            ? aText.localeCompare(bText, "fr")
            : bText.localeCompare(aText, "fr");
    });

    rows.forEach((row) => tbody.appendChild(row));
}
