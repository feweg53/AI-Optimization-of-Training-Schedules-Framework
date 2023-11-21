//script.js

document.getElementById("predictionForm").addEventListener("submit", function(event){
    event.preventDefault();

    let departureTime = document.getElementById("departureTime").value;
    let weatherCondition = document.getElementById("weatherCondition").value;
    let crewOnStandby = document.getElementById("crewOnStandby").value;

    let url = "http://localhost:5000/predict";

    fetch(url, {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json',
        },
        body: JSON.stringify({ 
            departureTime: departureTime, 
            weatherCondition: weatherCondition, 
            crewOnStandby: crewOnStandby 
        }),
    })
    .then(response => {
        if(response.ok) {
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
        document.getElementById("predictionResult").innerHTML = "Error in prediction: " + error;
    });
});
