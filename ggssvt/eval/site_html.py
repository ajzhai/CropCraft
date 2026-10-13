"""The project page, in the Nerfies template's layout.

Adapted from https://github.com/nerfies/nerfies.github.io (CC BY-SA 4.0). The
template's licence asks for a link back, which the footer carries.

Why that template and not something bespoke: it is the layout an academic reader
already knows how to scan, title, authors, a row of links, teaser, abstract,
then method and results, so nothing about the page needs learning before the
content does. It is also plain: Bulma's defaults, two Google fonts, black
buttons, white ground. Restraint is the point.

Output is a directory rather than one file, because that is what gets pushed to
GitHub Pages:

    site/
      index.html
      static/css/bulma.min.css
      static/css/site.css
      static/images/contact_sheet.png

The specimen payload stays inline in ``index.html``. It is base64 zlib and the
page is useless without it, so splitting it out would only add a fetch that can
fail under ``file://``.

Font Awesome is deliberately absent. The template loads it as 1.4 MB of
icons-via-JavaScript for what amounts to four glyphs; those are inlined as SVG
instead.
"""

from __future__ import annotations

import shutil
from pathlib import Path

from ..config import WORK_DIR

ASSETS = Path(__file__).parent / "assets"

# Four icons, taken as paths rather than pulling in an icon font for them.
ICONS = {
    "file": (
        "M14 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V8z M14 2v6h6"
    ),
    "code": "M16 18l6-6-6-6 M8 6l-6 6 6 6",
    "database": (
        "M12 2c4.4 0 8 1.3 8 3s-3.6 3-8 3-8-1.3-8-3 3.6-3 8-3z "
        "M20 5v14c0 1.7-3.6 3-8 3s-8-1.3-8-3V5 M20 12c0 1.7-3.6 3-8 3s-8-1.3-8-3"
    ),
    "github": (
        "M9 19c-5 1.5-5-2.5-7-3m14 6v-3.9a3.4 3.4 0 0 0-.9-2.6c3-.3 6.2-1.5 "
        "6.2-6.7A5.2 5.2 0 0 0 20 5.8 4.9 4.9 0 0 0 19.9 2s-1.1-.3-3.7 1.4a12.7 "
        "12.7 0 0 0-6.8 0C6.8 1.7 5.7 2 5.7 2A4.9 4.9 0 0 0 5.6 5.8 5.2 5.2 0 0 "
        "0 4.2 9.4c0 5.2 3.2 6.4 6.2 6.7a3.4 3.4 0 0 0-.9 2.5V23"
    ),
}


def _icon(name: str) -> str:
    return (
        '<span class="icon"><svg viewBox="0 0 24 24" fill="none" '
        'stroke="currentColor" stroke-width="2" stroke-linecap="round" '
        f'stroke-linejoin="round"><path d="{ICONS[name]}"/></svg></span>'
    )


def _link(href: str, icon: str, label: str) -> str:
    return (
        '<span class="link-block">'
        f'<a href="{href}" class="external-link button is-normal is-rounded is-dark">'
        f"{_icon(icon)}<span>{label}</span></a></span>"
    )


HEAD = """<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <meta name="description" content="Reconstructing above-ground plant biomass from
        twelve-view RGB-D capture, and measuring where the reconstruction fails.">
  <meta name="keywords" content="plant phenotyping, biomass, visual hull, RGB-D, GG-SSVT">
  <title>Automated Biomass Estimation</title>

  <link href="https://fonts.googleapis.com/css?family=Google+Sans|Noto+Sans|Castoro"
        rel="stylesheet">
  <link rel="stylesheet" href="./static/css/bulma.min.css">
  <link rel="stylesheet" href="./static/css/site.css">
</head>
<body>
"""


HERO = """
<section class="hero">
  <div class="hero-body">
    <div class="container is-max-desktop">
      <div class="columns is-centered">
        <div class="column has-text-centered">
          <h1 class="title is-2 publication-title">Automated Biomass Estimation<br>Using Self-Supervised Vision Transformers</h1>
          <div class="is-size-5 publication-authors">
            <span class="author-block">Aaron Masuba</span>
          </div>
          <div class="is-size-6 publication-authors mt-2">
            <span class="author-block">Supervisor: Prof. Herman Myburgh</span><br>
            <span class="author-block">Co-supervisors: Prof. Allan De Freitas
              &middot; Dr Kealeboga Mokise</span>
          </div>
          <div class="is-size-6 publication-authors mt-3">
            <span class="author-block">Smart Sensing and Intelligent Systems Group</span><br>
            <span class="author-block">Department of Electrical, Electronic and Computer
              Engineering, University of Pretoria</span>
          </div>
          <p class="publication-venue mt-3">MEng dissertation &middot; work in progress</p>

          <div class="column has-text-centered">
            <div class="publication-links">
              __LINKS__
            </div>
          </div>

          <p class="is-size-7 has-text-grey mt-4">
            Every figure on this page is generated from the pipeline, not drawn by hand.
            Results that did not survive checking are marked as withdrawn rather than removed.
          </p>
        </div>
      </div>
    </div>
  </div>
</section>

<section class="hero teaser">
  <div class="container is-max-desktop">
    <div class="hero-body">
      <img src="./static/images/contact_sheet.png" alt="Carved reconstructions of every specimen">
      <h2 class="subtitle has-text-centered caption">
        All __N__ specimens that pass the quality gate, carved from twelve RGB-D views each.
        The V batch, eight Eucalyptus in 17&ndash;32&nbsp;kg pots, is the most
        recent addition and the one that changed the conclusions.
      </h2>
    </div>
  </div>
</section>


<section class="section">
  <div class="container is-max-desktop">
    <div class="columns is-centered has-text-centered">
      <div class="column is-four-fifths">
        <h2 class="title is-3">Abstract</h2>
        <div class="content has-text-justified">
          <p>
            The question is whether the feed-forward pointmap family of reconstruction
            models, DUSt3R, MASt3R and Fast3R, improves both the accuracy of biomass
            estimation and what can be said about plant morphology from it. Answering that
            needs a reconstruction baseline to measure against, and most of what follows is
            the finding that the obvious baseline, space carving, is the wrong instrument.
          </p>
          <p>
            Thirty-eight potted plants, Eucalyptus saplings and Mango seedlings, were captured with two Kinect v2 units carried through six positions, giving twelve
            registered RGB-D views each. No calibration target was ever recorded, so every camera
            pose here is estimated from the depth data itself. Space carving turns those views
            into a 128&sup3; occupancy field, from which volume, height and shape descriptors are
            read and regressed against weighed above-ground mass.
          </p>
          <p>
            The first result is a measurement rather than a comparison. Dividing each weighed
            mass by its reconstructed above-ground volume gives an implied bulk density, which
            can be checked against physics: fresh plant tissue runs 300&ndash;900&nbsp;kg/m&sup3;.
            Only <strong>__PLAUSIBLE__ of __N__</strong> carved specimens land inside a
            deliberately generous 200&ndash;1000 band. Most fall far below it, because a visual
            hull encloses the air between leaves and branches. What is being measured is the
            canopy envelope, not the plant, and that is a property of silhouette carving rather
            than of resolution: the hull is the maximal solid consistent with the silhouettes,
            so finer voxels give a smoother envelope and never a gap.
          </p>
          <p>
            The second follows from it. The same twelve depth maps, integrated as a signed
            distance field instead of intersected as silhouette cones, put <strong>21 of 36</strong>
            reconstructions inside the band at the identical grid, and 31 at the resolution the
            sensor actually supports. Holding the features, the protocol, the grid, the views and
            the masks fixed and changing only that operator moves biomass RMSE from
            <strong>0.544 to 0.335&nbsp;kg</strong>, R&sup2; from +0.030 to +0.632, with a paired
            bootstrap of &minus;0.209 [&minus;0.363, &minus;0.066]. It is the first resolved
            improvement here, and it says the reconstruction was the bottleneck rather than the
            regressor.
          </p>
        </div>
        <div class="keyfacts">__KEYFACTS__</div>
      </div>
    </div>
  </div>
</section>
"""


BODY = """
<section class="section pt-0">
  <div class="container is-max-desktop">
    <h2 class="title is-3">Method</h2>
    <p class="mb-4 has-text-grey">Six stages, each one auditable on its own.</p>
    <div class="stages" id="pipe"></div>

    <h3 class="title is-5 mt-6">Every methodology, drawn</h3>
    <p class="is-size-7 has-text-grey mb-3">
      Each arm of this project as its own flow, drawn from the configuration and
      the measured results rather than by hand, so a figure cannot drift out of
      date the way a hand-made one does. The citations further down name these
      stages, so this is where to look up the one a reference points at.
    </p>
    <div class="field is-grouped is-grouped-multiline mb-3">
      <div class="control">
        <label class="label is-small" for="archpick">Methodology</label>
        <div class="select is-small"><select id="archpick"></select></div>
      </div>
    </div>
    <p class="is-size-7 has-text-grey mb-2" id="archnote"></p>
    <div class="tablewrap"><img id="archimg" alt="methodology diagram"
         style="width:100%;max-width:900px;display:block"></div>
    <p class="is-size-7 mt-3" id="archoutcome"></p>
  </div>
</section>


<section class="section">
  <div class="container is-max-desktop">
    <h2 class="title is-3">Every reconstruction</h2>
    <p class="mb-4 has-text-grey">
      Drag to rotate, scroll to zoom. The coloured dot beside each specimen carries its
      implied-density verdict: green is physically plausible, amber means the hull is an
      envelope, red means almost nothing was reconstructed.
    </p>

    <div class="controls">
      <div id="segmenter"></div>
      <div class="methodgroup">
        <span class="grouplabel">Colour</span>
        <div class="buttons has-addons" id="colour">
          <button class="button is-small" data-mode="segment" aria-pressed="true">Pot vs plant</button>
          <button class="button is-small" data-mode="z" aria-pressed="false">Height</button>
          <button class="button is-small" data-mode="depth" aria-pressed="false">Depth</button>
          <button class="button is-small" data-mode="solid" aria-pressed="false">Flat</button>
        </div>
      </div>
      <div class="methodgroup">
        <span class="grouplabel">Palette</span>
        <div class="select is-small">
          <select id="viewercmap">
            <option value="viridis" selected>viridis</option>
            <option value="plasma">plasma</option>
            <option value="greys">greys, for print</option>
          </select>
        </div>
      </div>
      <div class="methodgroup">
        <span class="grouplabel">Point size</span>
        <div class="select is-small">
          <select id="viewerdot">
            <option value="1">1 px</option>
            <option value="2">2 px</option>
            <option value="3" selected>3 px</option>
            <option value="4">4 px</option>
            <option value="5">5 px</option>
            <option value="7">7 px</option>
          </select>
        </div>
      </div>
      <span class="is-size-7 has-text-grey" id="cloudinfo"></span>
    </div>
    <p class="is-size-7 has-text-grey mb-4" id="methodnote"></p>

    <div class="viewer">
      <div>
        <div class="stagebox">
          <canvas id="stage"></canvas>
          <div class="legend" id="legend"></div>
        </div>
        <div class="level is-mobile mt-3 mb-0">
          <div class="level-left">
            <div class="level-item">
              <h3 class="title is-4 mb-0" id="sid">&nbsp;</h3>
              <span class="tag is-light ml-2" id="sspecies"></span>
            </div>
          </div>
          <div class="level-right">
            <span class="is-size-7 has-text-grey" id="ssub"></span>
          </div>
        </div>
        <div class="facts" id="sfacts"></div>
      </div>
      <div class="specimens" id="list"></div>
    </div>

    <h3 class="title is-5 mt-6" id="strip-heading">How this one was built</h3>
    <p class="is-size-7 has-text-grey mb-3">
      The same specimen, from the frames the cameras returned to the surface the
      mesh describes. A volume and its number cannot show where a reconstruction
      went wrong; these can. On the specimens raised on an upturned pot the
      staging is plain in the first row and invisible in every number on this
      page. Tiles load only for the specimen you select.
    </p>

    <div class="field is-grouped is-grouped-multiline mb-4" id="stripcontrols">
      <div class="control">
        <label class="label is-small" for="stripseg">Segmentation</label>
        <div class="select is-small"><select id="stripseg"></select></div>
      </div>
      <div class="control">
        <label class="label is-small" for="stripview">Turntable shows</label>
        <div class="select is-small">
          <select id="stripview">
            <option value="measured" selected>the measured points</option>
            <option value="shaded">the mesh, lit</option>
          </select>
        </div>
      </div>
    </div>
    <p class="is-size-7 has-text-grey mb-3" id="stripsegnote"></p>

    <div id="filmstrip"></div>

    <h3 class="title is-5 mt-6">Horizontal slices</h3>
    <p class="is-size-7 has-text-grey mb-3">
      The carve cut into six height bands, so the pot/canopy boundary is visible rather
      than asserted. The rim is estimated per specimen: pot mass spans 0.7 to 32&nbsp;kg
      across the batches, and one cut height cannot be right for all of them.
    </p>
    <div class="slices" id="slices"></div>
  </div>
</section>


<section class="section">
  <div class="container is-max-desktop">
    <h2 class="title is-3">Above-ground biomass</h2>
    <p class="mb-4 has-text-grey">
      Leave-one-out over every usable specimen, identical protocol for each method.
      Differences carry a paired bootstrap interval, because at this sample size a
      difference in means without one is not a result.
    </p>
    <div class="tablewrap mb-5"><table class="data" id="methods"></table></div>

    <h3 class="title is-5 mt-6">Changing the reconstruction, not the regressor</h3>
    <p class="is-size-7 has-text-grey mb-3">
      Identical features, protocol, grid, views and masks. The only difference is whether the
      occupancy came from intersecting silhouette cones or from integrating depth. Direct 2D
      touches no reconstruction, so it is the control and must not move.
    </p>
    <div class="tablewrap">
      <table class="data">
        <thead><tr><th>Method</th><th>carved</th><th>fused</th><th>paired bootstrap</th></tr></thead>
        <tbody>
          <tr class="best"><td>geometric features</td><td>0.544 / +0.030</td>
            <td><strong>0.335 / +0.632</strong></td>
            <td>&minus;0.209 [&minus;0.363, &minus;0.066]
              <span class="tag is-small is-success">resolved</span></td></tr>
          <tr><td>volume allometric</td><td>0.592 / &minus;0.150</td><td>0.469 / +0.278</td>
            <td>&minus;0.123 [&minus;0.202, &minus;0.034]
              <span class="tag is-small is-success">resolved</span></td></tr>
          <tr><td>canopy area allometric</td><td>0.598 / &minus;0.170</td><td>0.494 / +0.201</td>
            <td class="has-text-grey">clears the mean floor</td></tr>
          <tr><td>mesh geometry</td><td>0.507 / +0.157</td><td>0.486 / +0.227</td>
            <td class="has-text-grey"></td></tr>
          <tr><td>direct 2D</td><td>0.469 / +0.279</td><td>0.469 / +0.279</td>
            <td class="has-text-grey">control, unchanged</td></tr>
          <tr><td>mean predictor</td><td>0.568</td><td>0.568</td>
            <td class="has-text-grey"></td></tr>
        </tbody>
      </table>
    </div>
    <p class="is-size-7 has-text-grey mt-3">
      Volume allometry and canopy area both sat below the mean-predictor floor on the hull and
      clear it on the fusion. A single volume-to-mass law was said to be impossible across
      morphologies because hull density varied tenfold between a bushy Mango and a thin
      Eucalyptus; on a fused reconstruction the same law works, because the volumes are no
      longer envelopes of wildly different emptiness.
    </p>

    <div class="columns is-vcentered">
      <div class="column is-two-thirds">
        <canvas id="scatter" width="620" height="620" style="width:100%;height:auto"></canvas>
      </div>
      <div class="column">
        <h3 class="title is-5">Predicted against weighed</h3>
        <p class="is-size-7 has-text-grey">
          The dashed line is where a perfect method would sit. Hover a point for the
          specimen. Spread about the line is the error the table summarises.
        </p>
        <div id="scatterinfo"></div>
      </div>
    </div>
  </div>
</section>


<section class="section" id="backbone">
  <div class="container is-max-desktop">
    <h2 class="title is-3">Does a stronger backbone help?</h2>
    <p class="content">
      DINOv3 was gated for most of this project, which made it the obvious
      suspect for the gap between what the method should do and what it does.
      Access was granted and the answer is no. Read against the no-DINO control
      the two backbones are indistinguishable, and read against each other they
      are closer still: the interval is tight enough to rule out a difference
      worth chasing, rather than merely too wide to see one.
    </p>

    <div id="backbonefacts" class="keyfacts mb-4"></div>

    <p class="tablecap" id="probetablecap"></p>
    <div class="tablewrap">
      <table class="data booktabs" id="probetable"></table>
    </div>
    <p class="tablenote" id="probetablenote"></p>

    <h3 class="title is-5 mt-6">What each one sees</h3>
    <p class="is-size-7 has-text-grey mb-3">
      Patch features projected onto their own principal components and read as
      RGB. The projection is fitted on the subject rather than on the whole
      frame: these captures are a lit floor against a dark surround, and a fit
      over everything spends its first component on that and returns the plant
      as a smooth gradient. Read the boundaries the features draw, not the
      colours they take, which are arbitrary and only sign-aligned between the
      two panels.
    </p>
    <div class="field is-grouped is-grouped-multiline mb-3">
      <div class="control">
        <label class="label is-small" for="bbpick">Specimen</label>
        <div class="select is-small"><select id="bbpick"></select></div>
      </div>
      <div class="control">
        <label class="label is-small" for="bbview">Side</label>
        <div class="select is-small"><select id="bbview"></select></div>
      </div>
    </div>
    <div class="columns" id="backbonepanels"></div>
    <p class="is-size-7 has-text-grey" id="backbonenote"></p>
  </div>
</section>


<section class="section" id="stages">
  <div class="container is-max-desktop">
    <h2 class="title is-3">One specimen, every stage</h2>
    <p class="content">
      The five panels are not five parallel methods, and the row reads left to
      right for a reason. <strong>Segmentation is the shared input</strong>: the
      subject masks and their depth from twelve views, back-projected into the
      world. Everything to its right consumes that and nothing else.
      <strong>Silhouette carving and depth fusion are two genuinely different
      operators</strong> on it, and they differ in what they are willing to
      assume: carving keeps a voxel unless enough cameras vote it away, so it
      fills the space between leaves and comes out too large, while fusion only
      accepts surface the depth sensor actually returned. <strong>The mesh is not
      a third operator</strong>. It is marching cubes over the carve, so it is the
      same object drawn as a shell rather than a solid and can never contain
      anything the carve discarded, which is why its panel reaches the same
      height. <strong>The surface count, after
      <a href="#ledger">Nombambela (2025)</a>, is the fifth</strong>: no carving
      and no distance field, just the occupied voxels the measured surface passes
      through at 7 mm, from four of the same twelve views. At twelve views it
      would be the first panel counted differently, which is the clearest
      statement of what that operator is.
    </p>
    <p class="content">
      They share one camera, one voxel grid and one vertical scale, so a
      reconstruction that reaches lower is drawn lower. Each panel gives the
      estimate produced by the method that consumes that stage, except the
      surface count, which has no fitted regressor here and reports its implied
      bulk density and whether that clears the screen. Pick a specimen and drag
      any panel to turn all of them. Shift-scroll or the buttons zoom; a plain
      scroll moves the page.
    </p>

    <div class="field is-horizontal mb-3">
      <div class="field-label is-normal" style="flex-grow:0;margin-right:.75rem">
        <label class="label" for="stagepick">Specimen</label>
      </div>
      <div class="field-body">
        <div class="select is-small">
          <select id="stagepick"></select>
        </div>
        <div class="buttons are-small ml-3" id="stagezoom">
          <button class="button is-small" data-zoom="1.25">Zoom in</button>
          <button class="button is-small" data-zoom="0.8">Zoom out</button>
          <button class="button is-small" data-zoom="reset">Reset view</button>
        </div>
      </div>
    </div>

    <!-- Display only. These change nothing that is measured, and the point of
         offering them is that the panels are capped at 2500 points: at 2 px a
         canopy reads as a cloud and at 5 px as a solid, and which one you want
         depends on whether you are looking for shape or for extent. -->
    <div class="field is-grouped is-grouped-multiline mb-4" id="stagedisplay">
      <div class="control">
        <label class="label is-small" for="stagedot">Point size</label>
        <div class="select is-small">
          <select id="stagedot">
            <option value="1">1 px, finest</option>
            <option value="2">2 px</option>
            <option value="3" selected>3 px</option>
            <option value="4">4 px</option>
            <option value="5">5 px</option>
            <option value="7">7 px, solid</option>
          </select>
        </div>
      </div>
      <div class="control">
        <label class="label is-small" for="stagemode">Colour by</label>
        <div class="select is-small">
          <select id="stagemode">
            <option value="z" selected>height above the floor</option>
            <option value="depth">depth from the camera</option>
            <option value="segment">pot and plant</option>
            <option value="solid">one colour</option>
          </select>
        </div>
      </div>
      <div class="control">
        <label class="label is-small" for="stagecmap">Palette</label>
        <div class="select is-small">
          <select id="stagecmap">
            <option value="viridis" selected>viridis</option>
            <option value="plasma">plasma</option>
            <option value="greys">greys, for print</option>
          </select>
        </div>
      </div>
    </div>

    <div class="columns is-multiline" id="stagepanels"></div>

    <p class="is-size-7 has-text-grey" id="stagenote"></p>

    <article class="message is-info mt-3" id="stagegap">
      <div class="message-header"><p>Read the first two panels against each other</p></div>
      <div class="message-body">
        <p id="stagegapbody"></p>
      </div>
    </article>
  </div>
</section>


<section class="section" id="pedestal">
  <div class="container is-max-desktop">
    <h2 class="title is-3">Plant the carve threw away</h2>
    <p class="content">
      Most of these specimens were staged on an inverted pot, with the plant in a
      bag standing on top of it. The camera photographed the plant and the
      segmenter found it. The carve then stopped at the top of the stand and kept
      almost none of it, so the reported volume for those captures is furniture.
    </p>

    <div id="pedestalfacts" class="keyfacts mb-4"></div>

    <p class="tablecap" id="pedestaltablecap"></p>
    <div class="tablewrap">
      <table class="data booktabs" id="pedestaltable"></table>
    </div>
    <p class="tablenote" id="pedestaltablenote"></p>

    <article class="message is-danger">
      <div class="message-header"><p>Why a thin stem does not survive a carve</p></div>
      <div class="message-body">
        <p id="pedestalverdict"></p>
        <p class="is-size-7 mt-3">
          A stem a couple of centimetres across is thinner than a voxel, so most
          of the twelve cameras look straight past it and return the background
          behind, which votes the voxel empty. A voxel survives only when at most
          three of twelve dissent. Broad mango leaves clear that bar; eucalyptus
          seedlings do not. The loss is the operator&rsquo;s, not the capture&rsquo;s,
          which is why it is recoverable.
        </p>
      </div>
    </article>
  </div>
</section>


<section class="section" id="potmass">
  <div class="container is-max-desktop">
    <h2 class="title is-3">Can the pot masses be checked?</h2>
    <p class="content">
      Most of these pots were estimated rather than weighed, and the plant mass is
      the total minus that estimate, so a wrong estimate is a wrong target. Eleven
      pots were weighed, which gives something to calibrate against: a pot of damp
      medium has a real density, so the mass it was given and the volume the
      reconstruction puts below its rim have to agree.
    </p>

    <div id="potfacts" class="keyfacts mb-4"></div>

    <p class="tablecap" id="pottablecap"></p>
    <div class="tablewrap">
      <table class="data booktabs" id="pottable"></table>
    </div>
    <p class="tablenote" id="pottablenote"></p>

    <article class="message is-warning">
      <div class="message-header"><p>Sound where the geometry is clean, not identifiable where it is not</p></div>
      <div class="message-body">
        <p id="potverdict"></p>
        <p class="is-size-7 mt-3">
          The specimens that fail this check are the ones raised on a stand, and
          their below-rim hull contains that stand. It was never weighed, because
          it is not part of the specimen, so the low density says nothing about
          whether the estimate was right. Reverse estimating from a volume that
          holds unweighed furniture would replace one error with a larger one.
        </p>
      </div>
    </article>
  </div>
</section>


<section class="section" id="ledger">
  <div class="container is-max-desktop">
    <h2 class="title is-3">What settled, and what is still open</h2>
    <p class="content">
      The resolved results live in six different report files and the unresolved
      ones all live in one table, which gives the impression that nothing
      settled. Collected together the pattern is plain: paired counts and paired
      differences on large samples settle, and a difference in RMSE between two
      methods on thirty-six specimens does not.
    </p>

    <div id="ledgerfacts" class="keyfacts mb-4"></div>

    <div class="table-container">
      <table class="table is-fullwidth is-narrow" id="ledgertable"></table>
    </div>
  </div>
</section>


<section class="section" id="holdout">
  <div class="container is-max-desktop">
    <h2 class="title is-3">What the capture batch was worth to the score</h2>
    <p class="content">
      Leave-one-out withholds a specimen and leaves the other members of its own
      capture session in the training fold, carrying that session&rsquo;s mean mass.
      Leave-one-batch-out, LOBO, withholds the whole session. Everything else is
      held fixed, so the gap between the two columns is what the confound was
      worth. The transfer section below runs the same design on the lettuce set,
      holding out a cultivar instead of a capture batch.
    </p>

    <div class="table-container">
      <table class="table is-fullwidth is-narrow" id="holdouttable"></table>
    </div>

    <article class="message is-danger">
      <div class="message-header"><p>Read the last row first</p></div>
      <div class="message-body">
        <p id="holdoutverdict"></p>
        <p class="is-size-7 mt-3">
          It uses no geometry, no image and no features. It predicts a specimen&rsquo;s
          mass as the mean of the rest of its own capture batch, and under
          leave-one-out that beats every real method here. Under
          leave-one-batch-out every method falls below the mean predictor.
        </p>
      </div>
    </article>
  </div>
</section>


<section class="section" id="external">
  <div class="container is-max-desktop">
    <h2 class="title is-3">Does any of it transfer?</h2>
    <p class="content">
      The confound above cannot be fixed by re-analysis and there will be no
      further capture, so the question moved to somebody else&rsquo;s plants. The
      <a href="https://data.4tu.nl/articles/dataset/3rd_Autonomous_Greenhouse_Challenge_Online_Challenge_Lettuce_Images/15023088">4TU
      greenhouse lettuce set</a> is 387 usable RGB-D pairs across four cultivars
      and a seven-week growth series, destructively weighed, so its mass range is
      continuous by construction rather than clustered into sessions.
    </p>

    <div id="externalfacts" class="keyfacts mb-4"></div>

    <h3 class="title is-5">First, does the measurement work at all?</h3>
    <p class="content is-size-7">
      Their height, diameter and leaf area were measured destructively on the same
      plants, so the depth-derived versions can be checked against a ruler before
      anything is regressed. A pipeline whose diameter does not track a ruler has
      no business predicting mass.
    </p>
    <div class="table-container">
      <table class="table is-fullwidth is-narrow" id="externalchecks"></table>
    </div>

    <h3 class="title is-5">Then the regression, on a cultivar it never saw</h3>
    <p class="content is-size-7">
      This is the same LOBO design as the section above, with cultivar as the
      grouping variable in place of capture batch: the fit is scored on a variety
      it has never seen. The unscreened column is there because the screen uses
      their measured diameter, which correlates with mass, so a screened score is
      selected partly on the label.
    </p>
    <div class="table-container">
      <table class="table is-fullwidth is-narrow" id="externaltable"></table>
    </div>

    <article class="message is-success">
      <div class="message-header"><p>It transfers, and 3D earns its place</p></div>
      <div class="message-body">
        <p id="externalverdict"></p>
        <p class="is-size-7 mt-3">
          One top-down view cannot be carved or fused, so only the image-only half
          of the pipeline runs here. The surface descriptors come from
          back-projecting the depth map into a metric point cloud: rugosity, leaf
          angle from the normals, hull volume, and how the height is distributed.
        </p>
      </div>
    </article>
  </div>
</section>


<section class="section" id="reconstruction">
  <div class="container is-max-desktop">
    <h2 class="title is-3">A plant whose true shape is known</h2>
    <p class="content">
      Every claim above rests on a criterion that stands in for reference geometry,
      because no laser scan of these specimens exists. This one does not.
      <a href="https://www.ipb.uni-bonn.de/data/pheno4d/">Pheno4D</a> supplies
      fourteen laser-scanned plants; twelve virtual views of each were rendered at
      the rig&rsquo;s azimuths and put through the same carve and the same fusion.
      Drag any panel to turn all three together, shift-scroll or use the buttons
      to zoom, and pick a plant below.
    </p>

    <div class="field is-grouped is-grouped-multiline mb-3">
      <div class="buttons are-small" id="reconstruction-plants"></div>
    </div>

    <div class="buttons are-small mb-3" id="reconstruction-zoom">
      <button class="button is-small" data-zoom="1.25">Zoom in</button>
      <button class="button is-small" data-zoom="0.8">Zoom out</button>
      <button class="button is-small" data-zoom="reset">Reset view</button>
    </div>

    <div class="columns" id="reconstruction-panels"></div>

    <p class="is-size-7 has-text-grey" id="reconstruction-note"></p>

    <p class="tablecap" id="disttablecap"></p>
    <div class="tablewrap">
      <table class="data booktabs" id="disttable"></table>
    </div>
    <p class="tablenote" id="disttablenote"></p>

    <article class="message is-warning mt-4">
      <div class="message-header"><p>The usual metric ranks these backwards</p></div>
      <div class="message-body">
        <p id="reconstruction-inversion"></p>
        <p class="is-size-7 mt-3">
          A visual hull agrees with the silhouettes it was carved from
          <em>by construction</em>. Reprojecting it therefore measures whether the carve
          executed, never whether the shape is right, and a plant is mostly the gaps
          between its leaves, which no view ever sees through.
        </p>
      </div>
    </article>
  </div>
</section>


<section class="section">
  <div class="container is-max-desktop">
    <h2 class="title is-3">What these numbers cannot say</h2>

    <article class="message is-warning">
      <div class="message-header"><p>Most reconstructions cannot weigh what the plant weighs</p></div>
      <div class="message-body">
        <p id="plausibility"></p>
        <p class="is-size-7 mt-3">
          A visual hull encloses the space <em>between</em> leaves and branches, so for a canopy
          it measures the envelope rather than the plant. That is a property of the method at this
          resolution, not a fitting problem: no regressor recovers mass from a volume an
          order of magnitude too large.
        </p>
      </div>
    </article>

    <article class="message is-warning">
      <div class="message-header"><p>Batch membership explains more than any method</p></div>
      <div class="message-body">
        <p id="confound"></p>
        <p class="is-size-7 mt-3">
          So the comparison partly measures how well a method separates <em>size classes</em>,
          rather than how well it estimates mass among comparable plants. V001&ndash;V008 was
          captured to break this. Its masses span both existing clusters instead of forming
          a third, and it did, though not all the way.
        </p>
      </div>
    </article>

    <article class="message is-dark">
      <div class="message-header"><p>Withdrawn: &ldquo;reconstruction beats pixels&rdquo;</p></div>
      <div class="message-body">
        <p>
          An earlier version of this page reported that 3D features beat image-only regression,
          0.397 against 0.440&nbsp;kg RMSE. Re-tested with a paired bootstrap that difference is
          <strong>&minus;0.043&nbsp;kg, 95% CI [&minus;0.168, +0.099]</strong>, never
          resolved. The point estimate now goes the other way and is equally unresolved, and it
          flips again under a different feature-whitening choice. At this sample size the
          comparison cannot be settled by RMSE differences in either direction.
        </p>
      </div>
    </article>

    <div class="content is-size-7 has-text-grey" id="notes"></div>

    <article class="message is-info mt-4">
      <div class="message-header"><p>What that table can and cannot show</p></div>
      <div class="message-body">
        <p id="methodsresolution"></p>
        <p class="is-size-7 mt-3">
          A null result means nothing until the design says what it could have
          detected. The last column is the smallest difference each comparison
          would find four times in five, computed from its own bootstrap interval.
        </p>
      </div>
    </article>
  </div>
</section>


<section class="section">
  <div class="container is-max-desktop">
    <h2 class="title is-3">Next: pose-free reconstruction as an independent check</h2>
    <div class="content">
      <p>
        Every camera pose here is estimated from depth, and the azimuth refinement saturates its
        search bound on almost every specimen, the least verified assumption in the
        pipeline. Two explanations fit the implied-density result and they call for different
        fixes: either the visual hull is the wrong instrument for these species at any pose, or
        the poses are bad enough to inflate the hull.
      </p>
      <p>
        DUSt3R, MASt3R and Fast3R estimate cameras <em>and</em> geometry from images alone, so
        they share no failure mode with the carve. If the implied densities stay one to two orders
        of magnitude low with independently estimated poses, the envelope argument closes.
      </p>
    </div>
    <div class="tablewrap"><table class="data" id="next"></table></div>
  </div>
</section>


<section class="section">
  <div class="container is-max-desktop">
    <h2 class="title is-3">Experiment log</h2>
    <p class="mb-4 has-text-grey">
      Generated from the work directory on every page build, so it records what has
      actually run rather than what was intended. A row marked stale produced its numbers
      before the specimen cache it was fitted to, which is worse than a missing row
      because it looks finished.
    </p>
    <div class="keyfacts mb-5" id="progressfacts"></div>
    <div class="tablewrap"><table class="data" id="progress"></table></div>
  </div>
</section>


<section class="section" id="references">
  <div class="container is-max-desktop">
    <h2 class="title is-3">What this borrows, and where</h2>
    <p class="content">
      Each work is listed against the stage it informs and what is actually
      taken from it. Two supply a method, one supplies a metric this project had
      been missing, and one is a precedent rather than a source: it reached two
      of the same evaluation decisions independently, which is worth more as
      corroboration than it would be as an authority.
    </p>
    <div id="reflist"></div>
  </div>
</section>


<section class="section" id="BibTeX">
  <div class="container is-max-desktop content">
    <h2 class="title is-3">BibTeX</h2>
    <pre><code>@mastersthesis{masuba2026biomass,
  author  = {Masuba, Aaron},
  title   = {Reconstructing Plant Biomass from Twelve-View RGB-D Capture},
  school  = {University of Pretoria},
  year    = {2026},
  note    = {Work in progress}
}</code></pre>
    <p class="is-size-7 has-text-grey">The works cited above, for pasting:</p>
    <pre><code id="refbibtex"></code></pre>
  </div>
</section>


<footer class="footer">
  <div class="container">
    <div class="columns is-centered">
      <div class="column is-8">
        <div class="content has-text-centered is-size-7">
          <p>
            Ground truth is as-collected fresh mass, not oven-dry above-ground biomass. Pot mass is
            measured for V001&ndash;V008 and estimated for the rest. Camera poses are estimated,
            not measured. Every number on this page is regenerated by
            <code>python -m ggssvt.cli dashboard</code>.
          </p>
          <p>
            This page borrows the layout of the
            <a href="https://github.com/nerfies/nerfies.github.io">Nerfies project page</a>,
            used under a
            <a rel="license" href="http://creativecommons.org/licenses/by-sa/4.0/">Creative Commons
            Attribution-ShareAlike 4.0 International License</a>.
          </p>
        </div>
      </div>
    </div>
  </div>
</footer>
"""


def _keyfacts(summary: dict) -> str:
    """The four numbers a reader should leave with."""
    pl = summary.get("plausibility", {})
    facts = [
        (str(summary["n_specimens"]), "specimens"),
        (str(summary["n_views"]), "views each"),
        (f"{pl.get('n_plausible', 0)}/{pl.get('n', 0)}", "physically plausible"),
        (str(summary.get("batch_confound_r2", ", ")), "R² from batch alone"),
    ]
    return "".join(
        f'<div class="keyfact"><div class="num">{value}</div>'
        f'<div class="lab">{label}</div></div>'
        for value, label in facts
    )


def build_site(
    payload_json: str,
    summary: dict,
    *,
    out_dir: Path = WORK_DIR / "site",
    teaser: Path | None = None,
) -> Path:
    """Write the project page and its assets as a GitHub Pages directory."""
    css_dir = out_dir / "static" / "css"
    img_dir = out_dir / "static" / "images"
    css_dir.mkdir(parents=True, exist_ok=True)
    img_dir.mkdir(parents=True, exist_ok=True)

    for name in ("bulma.min.css", "site.css"):
        shutil.copyfile(ASSETS / name, css_dir / name)

    if teaser is not None and teaser.exists():
        shutil.copyfile(teaser, img_dir / "contact_sheet.png")

    # The filmstrip tiles travel as files rather than as base64 in the payload.
    # Fourteen megabytes of frames would make the page unopenable; as files the
    # browser fetches only the specimen someone actually selected, about 400 KB.
    tiles = WORK_DIR / "reports" / "filmstrip"
    if tiles.is_dir():
        shutil.copytree(tiles, out_dir / "static" / "filmstrip", dirs_exist_ok=True)

    # The methodology diagrams, as SVG so the text in them stays selectable and
    # stays sharp when an examiner zooms in on one box.
    diagrams = WORK_DIR / "reports" / "architecture"
    if diagrams.is_dir():
        target = out_dir / "static" / "architecture"
        target.mkdir(parents=True, exist_ok=True)
        for svg in diagrams.glob("*.svg"):
            shutil.copyfile(svg, target / svg.name)

    # The backbone feature panels.
    panels = WORK_DIR / "reports" / "backbone_viz"
    if panels.is_dir():
        shutil.copytree(panels, out_dir / "static" / "backbone_viz",
                        dirs_exist_ok=True)

    links = "\n              ".join(
        [
            _link("https://github.com/amasuba-dev/CropCraft", "github", "Code"),
            _link("./static/images/contact_sheet.png", "file", "Contact sheet"),
            _link("#BibTeX", "code", "BibTeX"),
        ]
    )

    pl = summary.get("plausibility", {})
    hero = (
        HERO.replace("__LINKS__", links)
        .replace("__KEYFACTS__", _keyfacts(summary))
        .replace("__PLAUSIBLE__", str(pl.get("n_plausible", 0)))
        .replace("__N__", str(summary["n_specimens"]))
    )

    script = (ASSETS / "viewer.js.tmpl").read_text(encoding="utf-8")
    page = (
        HEAD
        + hero
        + BODY
        + "\n<script>\n"
        + script.replace("__PAYLOAD__", payload_json)
        + "\n</script>\n</body>\n</html>\n"
    )

    index = out_dir / "index.html"
    index.write_text(page, encoding="utf-8")
    return index


__all__ = ["build_site"]
