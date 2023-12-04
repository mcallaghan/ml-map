var _fs = require("fs");
var _apacheArrow = require("apache-arrow");
var _arquero = require("arquero");
$(document).ready(function ($) {
  var checkExist = setInterval(function () {
    if ($('.cbutton').length) {
      clearInterval(checkExist);
      $(".cbutton").on('click', function (e) {
        $(this).toggleClass("clicked");
      });
      $("#load").on('click', function (e) {
        $(this).addClass("hidden");
      });
    }
  }, 100); // check every 100ms
});

const pako = require('pako');
// import { Zstd } from 'numcodecs';
// import { readFile } from 'lz4-napi';
// import { createDecoderStream } from 'lz4';

// const LZ4 = require('./lz4.js');
// import { createDecoderStream } from './lz4.js';
// console.log(tableFromIPC)
// console.log(readFileSync)
// const arrow = readFileSync("data/data.feather");
// const table = tableFromIPC(arrow);
// // const table = await tableFromIPC(fetch("data/data.feather"));
// console.table(table.toArray());

let df = _arquero.table({});
const loadData = async () => {
  // fetch_arrowfile_from_zip("static/data.feather").then((v) => {
  //     console.log(v)
  //     debugger;
  // })
  for (let i = 0; i<5; i++) {
    var startTime = performance.now();
    const response = await fetch(`assets/data_${i}.json`);
    const jsonData = await response.arrayBuffer();
    const raw = JSON.parse(pako.inflate(jsonData, {
      to: 'string'
    }));

    var batch_df = (0, _arquero.fromJSON)(raw);
    if (i==0) {
      df = batch_df
    } else {
      df = df.concat(batch_df)
    }
    $("#load").click()

  }

  // tableFromIPC(fetch("static/data.arrow")).then((value) => {
  //   df = fromArrow(value);
  //   console.log("Really loaded")
  //   console.log(df)
  //   debugger;
  //   $("#load").removeClass("hidden")
  // });
};




loadData(df);

window.dash_clientside = Object.assign({}, window.dash_clientside, {
  clientside: {
    button_filter: function () {

      var startTime = performance.now();
      if (df === 1) {
        var fig = {
          'data': [],
          // 'data': data,
          'layout': {
            'yaxis': {
              'visible': false,
              'range': [-2.5, 18] // todo set this dynamically
            },

            'xaxis': {
              'visible': false,
              'range': [-6.2, 12] // todo set this dynamically
            }
          }
        };

        var endTime = performance.now();
        console.log(endTime - startTime);
        return [fig, 0];
      }
      const cdict = {
        '8 - 01. AFOLU': '#3366CC',
        '8 - 02. Buildings': '#DC3912',
        '8 - 03. Industry': '#FF9900',
        '8 - 04. Energy': '#109618',
        '8 - 05. Transport': '#990099',
        '8 - 06. Waste': '#0099C6',
        '8 - 15. Cross-sectoral': '#DD4477'
      };
      var ctx = dash_clientside.callback_context;
      let cnames = df._names;

      // Filter instrument types
      let cre = new RegExp("^4");
      let cols = cnames.filter(x => cre.test(x));
      var endTime = performance.now();
      console.log(endTime - startTime);
      let fcols = [];
      for (let i = 0; i < cols.length; i++) {
        if (arguments[i] % 2 == 0) {
          fcols.push(`d["${cols[i]}"]`);
        }
      }
      let sub_df = df;
      if (fcols.length != cols.length) {
        if (fcols.length == 0) {
          sub_df = sub_df.filter('d["x"]>9999');
        } else {
          sub_df = sub_df.filter(fcols.join(" | "));
        }
      }

      // Filter by sector
      cre = new RegExp("^8");
      cols = cnames.filter(x => cre.test(x));
      fcols = [];
      for (let i = 0; i < cols.length; i++) {
        if (arguments[i + 5] % 2 == 0) {
          fcols.push(`d["${cols[i]}"]`);
        }
      }
      if (fcols.length != cols.length) {
        if (fcols.length == 0) {
          sub_df = sub_df.filter('d["x"]>9999');
        } else {
          sub_df = sub_df.filter(fcols.join(" | "));
        }
      }

      // Filter by the IDs from the search (done server side)
      const search = arguments[arguments.length - 3];
      if (search["filter"] === true) {
        const fdf = (0, _arquero.fromJSON)({
          "idx": search["ids"]
        });
        sub_df = sub_df.join(fdf);
      }

      let s = 6;
      let o = 0.6;
      let xmax = df.rollup({
        x: d => _arquero.op.max(d.x)
      }).array("x")[0];
      let xmin = df.rollup({
        x: d => _arquero.op.min(d.x)
      }).array("x")[0];
      let ymax = df.rollup({
        x: d => _arquero.op.max(d.y)
      }).array("x")[0];
      let ymin = df.rollup({
        x: d => _arquero.op.min(d.y)
      }).array("x")[0];

      const rl = arguments[arguments.length - 2]
      if (rl) {
        if (rl['xaxis.range[0]']) {
          xmin = rl['xaxis.range[0]']
          xmax = rl['xaxis.range[1]']
          ymin = rl['yaxis.range[0]']
          ymax = rl['yaxis.range[1]']
          sub_df = sub_df.params({
            xmin: xmin,
            xmax: xmax,
            ymin: ymin,
            ymax: ymax
          }).filter((d,$) =>
            (d.x > $.xmin) & (d.x < $.xmax) &
            (d.y > $.ymin) & (d.y < $.ymax)
          )
        } else if (rl['xaxis.autorange']) {
          //pass
        } else if (rl['autosize']) {
          //pass
        } else {
          return window.dash_clientside.no_update
        }
      }

      if (sub_df.numRows() > 10000) {
        s = 2;
        o = 0.2;
      } else if (sub_df.numRows() > 1000) {
        s = 4;
        o = 0.4;
      }

      var fig = {
        'data': [{
          // 'x': sdata.map(x => x["x"]),
          // 'y': sdata.map(x => x["y"]),
          'x': sub_df.array("x"),
          'y': sub_df.array("y"),
          'marker': {
            'color': sub_df.array("sector").map(x => cdict[x]),
            'size': s,
            'opacity': o
          },
          'hoverinfo': "none", //sub_df.array("title"),
          'mode': 'markers',
          'type': 'scattergl'
        }],
        // 'data': data,
        'layout': {
          'yaxis': {
            'visible': false,
            'range': [ymin, ymax]
          },

          'xaxis': {
            'visible': false,
            'range': [xmin, xmax]
          },
          'margin': {
            'l': 10,
            'r': 10,
            'b': 10,
            't': 25,
          }
        }
      };

      var endTime = performance.now();
      console.log(endTime - startTime);
      return [fig, sub_df.numRows(), sub_df.array("idx")];
    },
    hover: function(hoverData, d) {
      if (hoverData===null | hoverData===undefined) {
        return [false, window.dash_clientside.no_update, window.dash_clientside.no_update]
      }
      var hover_data = hoverData["points"][0]
      var bbox = hover_data["bbox"]
      var num = hover_data["pointNumber"]
      var idx = d[num]
      var t = df.column("title").get(idx)
      return [true, bbox = bbox, t]
    }
  }
});
