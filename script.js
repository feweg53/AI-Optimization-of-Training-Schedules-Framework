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

function renderChart(data) {
    const ctx = document.getElementById('predictionChart').getContext('2d');
    const chart = new Chart(ctx, {
        // The type of chart: e.g., 'bar', 'line', etc.
        type: 'bar',

        // The data for this dataset
        data: {
            labels: ['Prediction'],
            datasets: [{
                label: 'Flight Delay (minutes)',
                backgroundColor: 'rgb(255, 99, 132)',
                borderColor: 'rgb(255, 99, 132)',
                data: [data.prediction] // assuming 'data.prediction' is a numerical value
            }]
        },

        // Configuration options
        options: {}
    });
}
