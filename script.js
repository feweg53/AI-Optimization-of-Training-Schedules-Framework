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
    const ctx = document.getElementById('predictionChart').getContext('2d');
    const predictionChart = new Chart(ctx, {
        type: 'bar',
        data: {
            labels: ['Flight 1', 'Flight 2', 'Flight 3'], // Replace with dynamic labels
            datasets: [{
                label: 'Predicted Delay in Minutes',
                data: [12, 19, 3], // Replace with dynamic data
                backgroundColor: [
                    'rgba(255, 99, 132, 0.2)',
                    'rgba(54, 162, 235, 0.2)',
                    'rgba(75, 192, 192, 0.2)'
                ],
                borderColor: [
                    'rgba(255, 99, 132, 1)',
                    'rgba(54, 162, 235, 1)',
                    'rgba(75, 192, 192, 1)'
                ],
                borderWidth: 1
            }]
        },
        options: {
            scales: {
                y: {
                    beginAtZero: true
                }
            }
        }
    });
});
