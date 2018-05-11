import {Component} from '@angular/core';
import {HttpClient} from '@angular/common/http';
import {Chart} from 'chart.js';

@Component({
  selector: 'app-root',
  templateUrl: './app.component.html',
  styleUrls: ['./app.component.css']
})
export class AppComponent {
  title = 'app';

  chartLr = [];
  chartArima = [];
  chartLstm = [];

  lrData: JSON;
  arimaData: JSON;
  lstmData: JSON;
  htmlToAdd = ""
  simulationState = false;
  simulationStatusRaport: JSON


  constructor(private httpClient: HttpClient) {
  }

  ngOnInit() {
  }

  // labels = [item1{text:"Predicted"}, "Expected"]

  createChart(chart, data, canvas) {
    this.lrData = data as JSON;
    console.log(this.lrData);
    let real = JSON.parse(data['realValues'])
    let predicted = JSON.parse(data['predictedValues'])
    let dates = []
    dates = JSON.parse(data['dates']).map(res => new Date(res).toLocaleString().split(',')[0])
    this.chartLr = new Chart(canvas, {
      type: 'line',
      data: {
        labels: dates,
        datasets: [
          {
            label: 'Expected',
            data: real,
            borderColor: '#3cba9f',
            fill: false
          },
          {
            label: 'Predicted',
            data: predicted,
            borderColor: '#ffcc00',
            fill: false
          },
        ]
      },
      options: {
        legend: {
          display: true,
          position: 'bottom'
        },
        scales: {
          xAxes: [{
            display: true
          }],
          yAxes: [{
            display: true
          }]
        }
      }
    })
  }


  runSimulations() {
    this.httpClient.get('http://127.0.0.1:5002/forecast_price').subscribe(data => {
      this.simulationStatusRaport = data as JSON
      if (this.simulationStatusRaport['arima'] && this.simulationStatusRaport['lr'] && this.simulationStatusRaport['lstm']) {
        this.htmlToAdd = "<p>Simulations completed successfully</p>"
        this.simulationState = true

      }
      else {
        let message = "<p>Simulations failed, statuses: <br/>" +
          "Linear Regression: " + this.simulationStatusRaport['lr'] + "<br/>" +
          "ARIMA:" + this.simulationStatusRaport['arima'] + "<br/>" +
          "LSTM: " + this.simulationStatusRaport['lstm'] + "<br/></p>"

        this.htmlToAdd = message

      }
    })
  }


  getLinearRegressionData() {
    this.httpClient.get('http://127.0.0.1:5002/linear_regression').subscribe(data => {
      this.lrData = data as JSON;
      console.log(this.lrData);
      this.createChart(this.chartLr, data, "canvasLr")

    })
  }

  getArimaData() {
    this.httpClient.get('http://127.0.0.1:5002/arima').subscribe(data => {
      this.arimaData = data as JSON;
      console.log(this.arimaData);
      this.createChart(this.chartArima, data, "canvasArima")
    })
  }

  getLstmData() {
    this.httpClient.get('http://127.0.0.1:5002/lstm').subscribe(data => {
      this.lstmData = data as JSON;
      console.log(this.lstmData);
      this.createChart(this.chartLstm, data, "canvasLstm")
    })
  }
}
