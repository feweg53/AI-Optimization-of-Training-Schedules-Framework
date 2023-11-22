//script.js

document.getElementById("predictionForm").addEventListener("submit", function(event) {
    event.preventDefault();

    // Collect all necessary form inputs
    let formData = {
        departureTime: document.getElementById("departureTime").value,
        weatherCondition: document.getElementById("weatherCondition").value,
        crewOnStandby: document.getElementById("crewOnStandby").value,
        aircraftType: document.getElementById("aircraftType").value,
        destinationAirport: document.getElementById("destinationAirport").value,
        departureAirport: document.getElementById("departureAirport").value,
        departureCityCode: document.getElementById("departureCityCode").value,
        fullRouting: document.getElementById("fullRouting").value,
        generalEquipmentType: document.getElementById("generalEquipmentType").value,
        elapsedJourneyTime: document.getElementById("elapsedJourneyTime").value,
        scheduleFingerprint: document.getElementById("scheduleFingerprint").value,
        passengerClasses: document.getElementById("passengerClasses").value,
        // ... continue for all other fields
    };

    let url = "http://localhost:5000/predict";

    fetch(url, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(formData)
    })
    .then(response => {
        if (response.ok) {
            return response.json();
        } else {
            throw new Error('Server response was not OK');
        }
    })
    .then(data => {
        document.getElementById("predictionResult").innerHTML = "Prediction: " + data.prediction;
    })
    .catch((error) => {
        console.error('Error:', error);
        document.getElementById("predictionResult").innerHTML = "Error in prediction: " + error.message;
    });
});

document.addEventListener("DOMContentLoaded", function() {
    // Generate sample data for demonstration
    const sampleLabels = Array.from({length: 50}, (_, i) => `Data ${i+1}`);
    const sampleData = Array.from({length: 50}, () => Math.floor(Math.random() * 100));

    // Summarize data for pie chart
    const pieData = [sampleData.slice(0, 8).reduce((a, b) => a + b, 0),
                     sampleData.slice(8, 16).reduce((a, b) => a + b, 0),
                     sampleData.slice(16).reduce((a, b) => a + b, 0)];

    // Bar Chart
    const barCtx = document.getElementById('barChart').getContext('2d');
    const barChart = new Chart(barCtx, {
        type: 'bar',
        data: {
            labels: sampleLabels,
            datasets: [{
                label: 'Delayed Time (minute)',
                data: sampleData,
                backgroundColor: 'rgba(54, 162, 235, 0.2)',
                borderColor: 'rgba(54, 162, 235, 1)',
                borderWidth: 1
            }]
        },
        options: {
            scales: {
                yAxes: [{ ticks: { beginAtZero: true } }]
            }
        }
    });

    // Line Chart
    const lineCtx = document.getElementById('lineChart').getContext('2d');
    const lineChart = new Chart(lineCtx, {
        type: 'line',
        data: {
            labels: sampleLabels,
            datasets: [{
                label: 'Delayed Time (minute)',
                data: sampleData,
                backgroundColor: 'rgba(255, 99, 132, 0.2)',
                borderColor: 'rgba(255, 99, 132, 1)',
                borderWidth: 1
            }]
        },
        options: {
            scales: {
                yAxes: [{ ticks: { beginAtZero: true } }]
            }
        }
    });

    // Pie Chart
    const pieCtx = document.getElementById('pieChart').getContext('2d');
    const pieChart = new Chart(pieCtx, {
        type: 'pie',
        data: {
            labels: ['Late', 'Very Late', 'Extremely Late'],
            datasets: [{
                label: 'Delayed Time (minute)',
                data: pieData,
                backgroundColor: ['rgba(75, 192, 192, 0.5)', 'rgba(255, 205, 86, 0.5)', 'rgba(255, 99, 132, 0.5)'],
                borderColor: ['rgba(75, 192, 192, 1)', 'rgba(255, 205, 86, 1)', 'rgba(255, 99, 132, 1)'],
                borderWidth: 1
            }]
        },
        options: {
            responsive: false, //Careful! Making it true will likely cost an error!
            maintainAspectRatio: false
        }
    });
});

});
