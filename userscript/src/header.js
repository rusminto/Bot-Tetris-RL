// ==UserScript==
// @name         Tetris® AI AutoPlayer (CEM-RL Policy Search & DQN)
// @namespace    https://github.com/rusminto/Bot-Tetris-RL
// @version      __VERSION__
// @description  Reinforcement-learning Tetris bot (CEM policy search & DQN) for play.tetris.com and a local offline copy of the game
// @match        http://localhost:*/*
// @match        http://127.0.0.1:*/*
// @match        https://play.tetris.com/*
// @grant        none
// @run-at       document-start
// @allFrames    true
// ==/UserScript==

(function() {
    'use strict';

    if (window.__tetrisBotInitialized) return;
    window.__tetrisBotInitialized = true;

    const isTopFrame = (window.self === window.top);

    console.log(
        `%c[TetrisRL Bot] v__VERSION__ (CEM-RL Policy Search & DQN) loaded in ${isTopFrame ? 'TOP WINDOW' : 'GAME IFRAME'} (${location.href.slice(0, 60)}...)`,
        "color: #10b981; font-weight: bold; font-size: 13px;"
    );

