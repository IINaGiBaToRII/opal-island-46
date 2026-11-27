
use crate::geometry::{Diag10, Diag8, Diag8M, Geometry, Ortho8};

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum KingLanding {
    Any,
    Next,
}

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum CapturePriority {
    None,
    MaxCount,
    Italian,
    Spanish,
    Czech,
}

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum PromotionInCapture {
    ContinueAsKing,
    Stop,
    Transit,
}

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Removal {
    AfterMove,
    Immediate,
}

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Numbering {
    None,
    TopDown,
    BottomUpRightToLeft,
}

pub trait Rules: 'static + Copy + Clone + Default + Eq + Send + Sync + std::fmt::Debug {
    type Geo: Geometry;
    const NAME: &'static str;
    const FORCED: bool;
    const MEN_CAPTURE_BACKWARD: bool;
    const MAN_CAPTURES_KING: bool;
    const KING_FLYING: bool;
    const KING_LANDING: KingLanding;
    const KING_MUST_CONTINUE: bool;
    const PRIORITY: CapturePriority;
    const PROMOTION: PromotionInCapture;
    const REMOVAL: Removal;
    const NO_U_TURN: bool;
    const WHITE_FIRST: bool;
    const START_DFEN: &'static str;
    const KING_VALUE: i32;
    const DRAW_MOVES: u32;
    const ONE_V_ONE_DRAW: bool;
    const NUMBERING: Numbering;
    const NUMERIC_NOTATION: bool;
}

macro_rules! variant {
    (
        $(#[$m:meta])* $name:ident, $str:expr, $geo:ty,
        back = $back:expr, man_king = $mk:expr, flying = $fly:expr, landing = $land:ident, must_cont = $mc:expr,
        prio = $prio:ident, promo = $promo:ident, removal = $rem:ident, no_u = $nu:expr, white_first = $wf:expr,
        start = $start:expr, king_value = $kv:expr, draw = $draw:expr, one_v_one = $ovo:expr,
        numbering = $num:ident, numeric = $numeric:expr
    ) => {
        $(#[$m])*
        #[derive(Clone, Copy, Default, PartialEq, Eq, Debug)]
        pub struct $name<const FORCED: bool>;

        impl<const F: bool> Rules for $name<F> {
            type Geo = $geo;
            const NAME: &'static str = $str;
            const FORCED: bool = F;
            const MEN_CAPTURE_BACKWARD: bool = $back;
            const MAN_CAPTURES_KING: bool = $mk;
            const KING_FLYING: bool = $fly;
            const KING_LANDING: KingLanding = KingLanding::$land;
            const KING_MUST_CONTINUE: bool = $mc;
            const PRIORITY: CapturePriority = CapturePriority::$prio;
            const PROMOTION: PromotionInCapture = PromotionInCapture::$promo;
            const REMOVAL: Removal = Removal::$rem;
            const NO_U_TURN: bool = $nu;
            const WHITE_FIRST: bool = $wf;
            const START_DFEN: &'static str = $start;
            const KING_VALUE: i32 = $kv;
            const DRAW_MOVES: u32 = $draw;
            const ONE_V_ONE_DRAW: bool = $ovo;
            const NUMBERING: Numbering = Numbering::$num;
            const NUMERIC_NOTATION: bool = $numeric;
        }
    };
}

variant!(
    Russian, "russian", Diag8,
    back = true, man_king = true, flying = true, landing = Any, must_cont = true,
    prio = None, promo = ContinueAsKing, removal = AfterMove, no_u = false, white_first = true,
    start = "w .b.b.b.b/b.b.b.b./.b.b.b.b/......../......../w.w.w.w./.w.w.w.w/w.w.w.w.", king_value = 300, draw = 15, one_v_one = false,
    numbering = None, numeric = false
);

variant!(
    Brazilian, "brazilian", Diag8,
    back = true, man_king = true, flying = true, landing = Any, must_cont = false,
    prio = MaxCount, promo = Transit, removal = AfterMove, no_u = false, white_first = true,
    start = "w .b.b.b.b/b.b.b.b./.b.b.b.b/......../......../w.w.w.w./.w.w.w.w/w.w.w.w.", king_value = 300, draw = 20, one_v_one = false,
    numbering = None, numeric = false
);

variant!(
    International, "international", Diag10,
    back = true, man_king = true, flying = true, landing = Any, must_cont = false,
    prio = MaxCount, promo = Transit, removal = AfterMove, no_u = false, white_first = true,
    start = "w .b.b.b.b.b/b.b.b.b.b./.b.b.b.b.b/b.b.b.b.b./........../........../.w.w.w.w.w/w.w.w.w.w./.w.w.w.w.w/w.w.w.w.w.", king_value = 300, draw = 25, one_v_one = false,
    numbering = TopDown, numeric = true
);

variant!(
    American, "american", Diag8,
    back = false, man_king = true, flying = false, landing = Next, must_cont = false,
    prio = None, promo = Stop, removal = AfterMove, no_u = false, white_first = false,
    start = "b .b.b.b.b/b.b.b.b./.b.b.b.b/......../......../w.w.w.w./.w.w.w.w/w.w.w.w.", king_value = 170, draw = 40, one_v_one = false,
    numbering = TopDown, numeric = false
);

variant!(
    Italian, "italian", Diag8M,
    back = false, man_king = false, flying = false, landing = Next, must_cont = false,
    prio = Italian, promo = Stop, removal = AfterMove, no_u = false, white_first = true,
    start = "w b.b.b.b./.b.b.b.b/b.b.b.b./......../......../.w.w.w.w/w.w.w.w./.w.w.w.w", king_value = 170, draw = 40, one_v_one = false,
    numbering = TopDown, numeric = false
);

variant!(
    Spanish, "spanish", Diag8M,
    back = false, man_king = true, flying = true, landing = Any, must_cont = false,
    prio = Spanish, promo = Stop, removal = AfterMove, no_u = false, white_first = true,
    start = "w b.b.b.b./.b.b.b.b/b.b.b.b./......../......../.w.w.w.w/w.w.w.w./.w.w.w.w", king_value = 300, draw = 20, one_v_one = false,
    numbering = BottomUpRightToLeft, numeric = false
);

variant!(
    Czech, "czech", Diag8,
    back = false, man_king = true, flying = true, landing = Any, must_cont = false,
    prio = Czech, promo = Stop, removal = AfterMove, no_u = false, white_first = true,
    start = "w .b.b.b.b/b.b.b.b./.b.b.b.b/......../......../w.w.w.w./.w.w.w.w/w.w.w.w.", king_value = 300, draw = 15, one_v_one = false,
    numbering = None, numeric = false
);

variant!(
    Thai, "thai", Diag8,
    back = false, man_king = true, flying = true, landing = Next, must_cont = false,
    prio = None, promo = Stop, removal = Immediate, no_u = false, white_first = false,
    start = "b .b.b.b.b/b.b.b.b./......../......../......../......../.w.w.w.w/w.w.w.w.", king_value = 250, draw = 0, one_v_one = false,
    numbering = None, numeric = false
);

variant!(
    Turkish, "turkish", Ortho8,
    back = false, man_king = true, flying = true, landing = Any, must_cont = false,
    prio = MaxCount, promo = Transit, removal = Immediate, no_u = true, white_first = true,
    start = "w ......../bbbbbbbb/bbbbbbbb/......../......../wwwwwwww/wwwwwwww/........", king_value = 450, draw = 0, one_v_one = true,
    numbering = None, numeric = false
);
